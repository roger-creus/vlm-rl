import os
import random
import re
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from PIL import Image
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig
from accelerate.utils import TorchDynamoPlugin
from accelerate import Accelerator 

from src.models.model import Agent
from src.utils.args import Args
from src.utils.utils import numpy_to_pil, make_env, parse_action
from src.utils.action_maps import action_maps

from IPython import embed

if __name__ == "__main__":
    args = tyro.cli(Args)

    accelerator_cfg = {"gradient_accumulation_steps": 1}
    if args.enable_compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="reduce-overhead", # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=True,
            dynamic=False
        )
        accelerator_cfg["dynamo_plugin"] = dynamo_plugin
        print("Enabling compilation!")

    accelerator = Accelerator(**accelerator_cfg)
    device = accelerator.device

    args.total_batch_size = int(args.num_envs * args.num_steps * accelerator.num_processes)
    args.minibatch_size = int(args.total_batch_size // args.num_minibatches)
    per_process_minibatch_size = int(args.minibatch_size // accelerator.num_processes)
    args.num_iterations = args.total_timesteps // args.total_batch_size
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print(" VLM PPO Distributed Training Configuration")
        print("="*50)
        print(f" 🚀 Number of GPUs: {accelerator.num_processes}")
        print(f" 🌍 Environments per GPU: {args.num_envs}")
        print(f" 👣 Steps per Environment: {args.num_steps}")
        print("-" * 50)
        print(f" 📊 Total Batch Size (envs * steps * gpus): {args.total_batch_size}")
        print(f" 📦 Total Minibatches: {args.num_minibatches}")
        print(f"  -> Minibatch Size (total_batch / minibatches): {args.minibatch_size}")
        print(f"  -> Minibatch Size Per Worker (total_batch / minibatches / {accelerator.num_processes}): {per_process_minibatch_size}")
        print("-" * 50)
        print(f" 🎯 Total Timesteps: {args.total_timesteps:,}")
        print(f" 🔄 Total Training Iterations: {args.num_iterations:,}")
        print("="*50 + "\n")
    accelerator.wait_for_everyone()

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = per_process_minibatch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # --- Logging ---
    if accelerator.is_main_process:
        log_path = os.path.join(f"runs/{run_name}", args.log_dir)
        os.makedirs(log_path, exist_ok=True)
        interaction_log_file = open(os.path.join(log_path, "interactions.txt"), "w")

        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                config=vars(args), name=run_name, monitor_gym=True, save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))

    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # --- Per-Process Environments ---
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    action_map = action_maps[args.env_id]
    with open(args.prompt_path, "r") as f:
        prompt_text = f.read()

    # --- Agent and Optimizer ---
    agent = Agent(envs, args.vlm_name)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=agent.target_modules,
        )
        lora_layers = filter(lambda p: p.requires_grad, agent.model.parameters())
        critic_layers = filter(lambda p: p.requires_grad, agent.critic.parameters())
        parameters = list(lora_layers) + list(critic_layers)
    else:
        parameters = agent.parameters()
    
    optimizer = optim.Adam(parameters, lr=args.learning_rate, eps=1e-5)
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # --- Storage Tensors ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    prompt_lens = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    max_seq_len = 1024
    full_input_ids = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long, device=device)
    full_attention_masks = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, max_seq_len), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    for iteration in range(1, args.num_iterations + 1):
        #accelerator.unwrap_model(agent).eval()
        
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase: Each process collects its own data ---
        for step in range(0, args.num_steps):
            if accelerator.is_main_process:
                global_step += args.num_envs * accelerator.num_processes
            
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                logprob, value, f_ids, f_mask, p_len, generated_texts = agent.get_action_and_value(
                    obs=next_obs, 
                    text_prompts=[prompt_text] * args.num_envs,
                    action_ids=None,
                    prompt_lens=None,
                )
                
                action = torch.tensor(
                    [parse_action(text, envs.single_action_space, action_map) for text in generated_texts],
                    device=device
                )
                values[step] = value.flatten()
                            
            actions[step] = action
            logprobs[step, :, :logprob.shape[1]] = logprob
            prompt_lens[step] = p_len 
            
            seq_len = min(f_ids.shape[1], max_seq_len)
            if seq_len > max_seq_len:
                print(f"WARNING:seq_len > max_seq_len: {seq_len} > {max_seq_len}")
                
            full_input_ids[step, :, :seq_len] = f_ids[:, :seq_len]
            full_attention_masks[step, :, :seq_len] = f_mask[:, :seq_len]

            # --- Step ---
            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(term, trunc)
            print(f"🚀 [Process {accelerator.process_index}] Step {step+1}/{args.num_steps}")

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                    
            # --- Logging ---
            if accelerator.is_main_process:
                if iteration % args.log_every == 0 and step == 0:
                    pil_images_debug = numpy_to_pil(next_obs.cpu().numpy())
                    log_image = pil_images_debug[0]
                    image_path = os.path.join(log_path, f"iter_{iteration}_step_{step}.png")
                    log_image.save(image_path)
                    
                    log_entry = (
                        f"## Iteration {iteration}, Step {step} (Global Step: {global_step})\n\n"
                        f"**Prompt:**\n```\n{prompt_text}\n```\n\n"
                        f"**VLM Output (Env 0):**\n```json\n{generated_texts[0]}\n```\n\n"
                        f"**Observation (Env 0):**\n"
                        f"![Observation]({os.path.basename(image_path)})\n\n"
                        "---\n\n"
                    )
                    interaction_log_file.write(log_entry)
                    interaction_log_file.flush()
                    
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(obs=next_obs, prompt_text=prompt_text).flatten()

        # rewards is of shape (num_steps, num_envs) and when gathered it becomes (num_processes * num_steps, num_envs)
        # same with all other tensors
        # gather concatenates the tensors along the first dimension

        gathered_rewards = accelerator.gather(rewards) # shape (num_processes * num_steps, num_envs)
        gathered_values = accelerator.gather(values) # shape (num_processes * num_steps, num_envs)
        gathered_dones = accelerator.gather(dones) # shape (num_processes * num_steps, num_envs)
        gathered_next_value = accelerator.gather(next_value) # shape (num_processes * num_envs)
        gathered_next_done = accelerator.gather(next_done) # shape (num_processes * num_envs)

        num_total_envs = args.num_envs * accelerator.num_processes
        gathered_rewards = gathered_rewards.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # shape (num_steps, num_processes * num_envs)
        gathered_values = gathered_values.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # shape (num_steps, num_processes * num_envs)
        gathered_dones = gathered_dones.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # shape (num_steps, num_processes * num_envs)
        gathered_next_value = gathered_next_value.view(num_total_envs) # shape (num_processes * num_envs)       
        gathered_next_done = gathered_next_done.view(num_total_envs) # shape (num_processes * num_envs)

        with torch.no_grad():
            advantages = torch.zeros_like(gathered_rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - gathered_next_done
                    nextvalues = gathered_next_value
                else:
                    nextnonterminal = 1.0 - gathered_dones[t + 1]
                    nextvalues = gathered_values[t + 1]
                delta = gathered_rewards[t] + args.gamma * nextvalues * nextnonterminal - gathered_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + gathered_values

        gathered_obs = accelerator.gather(obs) # shape (num_processes * num_steps, num_envs, *obs_shape)
        gathered_logprobs = accelerator.gather(logprobs) # shape (num_processes * num_steps, num_envs)
        gathered_input_ids = accelerator.gather(full_input_ids) # shape (num_processes * num_steps, num_envs, max_seq_len)
        gathered_prompt_lens = accelerator.gather(prompt_lens) # shape (num_processes * num_steps, num_envs)
        gathered_attention_masks = accelerator.gather(full_attention_masks) # shape (num_processes * num_steps, num_envs, max_seq_len)

        b_logprobs = gathered_logprobs.view(
            accelerator.num_processes, args.num_steps, args.num_envs, max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, max_seq_len) # shape (num_processes * num_steps * num_envs, max_seq_len)

        b_input_ids = gathered_input_ids.view(
            accelerator.num_processes, args.num_steps, args.num_envs, max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, max_seq_len) # shape (num_processes * num_steps * num_envs, max_seq_len)

        b_prompt_lens = gathered_prompt_lens.view(
            accelerator.num_processes, args.num_steps, args.num_envs
        ).permute(1, 0, 2).reshape(-1)
        
        b_attention_masks = gathered_attention_masks.view(
            accelerator.num_processes, args.num_steps, args.num_envs, max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, max_seq_len)
        
        b_obs = gathered_obs.view(
            accelerator.num_processes, args.num_steps, args.num_envs, *envs.single_observation_space.shape
        ).permute(1, 0, 2, 3, 4, 5).reshape(-1, *envs.single_observation_space.shape)

        b_advantages = advantages.view(
            accelerator.num_processes, args.num_steps, args.num_envs
        ).permute(1, 0, 2).reshape(-1)
        
        b_returns = returns.view(
            accelerator.num_processes, args.num_steps, args.num_envs
        ).permute(1, 0, 2).reshape(-1)
        
        b_values = gathered_values.view(
            accelerator.num_processes, args.num_steps, args.num_envs
        ).permute(1, 0, 2).reshape(-1) # shape (num_processes * num_steps * num_envs)

        b_inds = np.arange(args.total_batch_size)
        clipfracs = []
        if accelerator.is_main_process:
            print("Training...")

        #accelerator.unwrap_model(agent).train()
        for epoch in range(args.update_epochs):
            epoch_clipfracs = []
            epoch_approx_kls = []

            np.random.shuffle(b_inds)
            for start in range(0, args.total_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                process_mb_inds = mb_inds[accelerator.process_index::accelerator.num_processes]
                print(f"[Process {accelerator.process_index}] {process_mb_inds}")
                
                with accelerator.accumulate(agent):
                    mb_obs = b_obs[process_mb_inds].to(device)
                    mb_input_ids = b_input_ids[process_mb_inds].to(device)
                    mb_prompt_lens = b_prompt_lens[process_mb_inds].to(device)
                    mb_attention_masks = b_attention_masks[process_mb_inds].to(device)
                    mb_logprobs = b_logprobs[process_mb_inds].to(device)
                    mb_advantages = b_advantages[process_mb_inds].to(device)
                    mb_returns = b_returns[process_mb_inds].to(device)
                    mb_values = b_values[process_mb_inds].to(device)

                    newlogprob, newvalue, entropy_tensor, final_mask_from_agent = agent.get_action_and_value(
                        obs=mb_obs,
                        text_prompts=[prompt_text] * mb_obs.shape[0],
                        action_ids=mb_input_ids,
                        prompt_lens=mb_prompt_lens
                    )
                    newvalue = newvalue.view(-1)
                    
                    true_final_mask = final_mask_from_agent & mb_attention_masks.bool()
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()
                    ratio = torch.where(true_final_mask, ratio, 1.0)
                    
                    with torch.no_grad():
                        valid_token_count = true_final_mask.sum()
                        if valid_token_count > 0:
                            masked_logratio = torch.where(true_final_mask, logratio, 0.0)
                            approx_kl = (((ratio - 1) - masked_logratio) * true_final_mask).sum() / valid_token_count
                            epoch_approx_kls.append(approx_kl.item())
                            
                            clipfrac = (torch.gt(torch.abs(ratio - 1.0), args.clip_coef).float() * true_final_mask).sum() / valid_token_count
                            epoch_clipfracs.append(clipfrac.item())

                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    mb_advantages = mb_advantages.unsqueeze(-1)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss_per_token = torch.max(pg_loss1, pg_loss2)
                    
                    pg_loss = (pg_loss_per_token * true_final_mask).sum() / true_final_mask.sum()
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    
                    entropy_loss = (entropy_tensor * true_final_mask).sum() / true_final_mask.sum()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

        # --- Logging ---
        if accelerator.is_main_process:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            
            if epoch_approx_kls:
                writer.add_scalar("losses/approx_kl", np.mean(epoch_approx_kls), global_step)
            if epoch_clipfracs:
                writer.add_scalar("losses/clipfrac", np.mean(epoch_clipfracs), global_step)
            
            sps = int(args.total_batch_size / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"SPS: {sps} || value.loss : {v_loss.item()}, policy.loss : {pg_loss.item()}, policy.entropy : {entropy_loss.item()}")

    envs.close()
    writer.close()