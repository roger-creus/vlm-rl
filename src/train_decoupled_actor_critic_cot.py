import os
import random
import time
import gymnasium as gym
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from accelerate.utils import TorchDynamoPlugin
from accelerate import Accelerator 
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO

from src.models.model import DecoupledActorCriticVLM_COT
from src.utils.args import Args
from src.utils.utils import numpy_to_pil, make_vizdoom_env, parse_action_cot, gc_cuda_cleanup, print_trainable_parameters
from src.utils.action_maps import action_maps

from IPython import embed

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"env={args.env_id}_nenvs={args.num_envs}_nsteps={args.num_steps}_epochs={args.update_epochs}_seed={args.seed}_time={int(time.time())}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    
    # --- Accelerator ---
    accelerator_cfg = {"project_dir": f"runs/{run_name}", "gradient_accumulation_steps": args.gradient_accumulation_steps}
    if args.enable_compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False
        )
        accelerator_cfg["dynamo_plugin"] = dynamo_plugin
        print("Compilation enabled!")
    
    accelerator = Accelerator(**accelerator_cfg)
    
    # --- Arguments ---
    device = accelerator.device
    args.total_batch_size = int(args.num_envs * args.num_steps * accelerator.num_processes)
    args.minibatch_size = int(args.total_batch_size // args.num_minibatches)
    per_process_minibatch_size = int(args.minibatch_size // accelerator.num_processes)
    args.num_iterations = args.total_timesteps // args.total_batch_size
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print(" VLM PPO Distributed Training Configuration")
        print("="*60)
        print(f" 🚀 Number of GPUs (num_gpus): {accelerator.num_processes}")
        print(f" 🌍 Environments per GPU (num_envs): {args.num_envs}")
        print(f" 👣 Steps per Environment (num_steps): {args.num_steps}")
        print("-" * 60)
        total_batch_formula = "num_envs * num_steps * num_gpus"
        total_batch_subst = f"{args.num_envs} * {args.num_steps} * {accelerator.num_processes}"
        print(f" 📊 Total Batch Size ({total_batch_formula}) = ({total_batch_subst}) = {args.total_batch_size}")
        print(f" 📦 Total Minibatches (num_minibatches): {args.num_minibatches}")
        minibatch_formula = "total_batch_size / num_minibatches"
        minibatch_subst = f"{args.total_batch_size} / {args.num_minibatches}"
        print(f"  -> Minibatch Size ({minibatch_formula}) = ({minibatch_subst}) = {args.minibatch_size}")
        minibatch_per_worker_formula = "minibatch_size / num_gpus"
        minibatch_per_worker_subst = f"{args.minibatch_size} / {accelerator.num_processes}"
        print(f"  -> Minibatch Size Per Worker ({minibatch_per_worker_formula}) = ({minibatch_per_worker_subst}) = {per_process_minibatch_size}")
        print(f"  -> Gradient Accumulation Steps (gradient_accumulation_steps): {args.gradient_accumulation_steps}")
        true_grad_steps_formula = "(num_minibatches / gradient_accumulation_steps) * update_epochs"
        true_grad_steps_subst = f"({args.num_minibatches} / {args.gradient_accumulation_steps}) * {args.update_epochs}"
        true_grad_steps = (args.num_minibatches / args.gradient_accumulation_steps) * args.update_epochs
        print(f"  -> True Gradient Steps ({true_grad_steps_formula}) = ({true_grad_steps_subst}) = {true_grad_steps}")
        print("-" * 60)
        print(f" 🎯 Total Timesteps (total_timesteps): {args.total_timesteps:,}")
        total_iter_formula = "total_timesteps / total_batch_size"
        total_iter_subst = f"{args.total_timesteps} / {args.total_batch_size}"
        print(f" 🔄 Total Training Iterations ({total_iter_formula}) = ({total_iter_subst}) = {args.num_iterations:,}")
        print("="*60 + "\n")
    accelerator.wait_for_everyone()

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = per_process_minibatch_size

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
        [make_vizdoom_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    envs.single_action_space = envs.envs[0].action_space
    envs.single_observation_space = envs.envs[0].observation_space
    action_map = action_maps[args.env_id]
    with open(args.prompt_actor_path, "r") as f:
        prompt_text_actor = f.read()
    with open(args.prompt_critic_path, "r") as f:
        prompt_text_critic = f.read()

    # --- Agent and Optimizer ---
    agent = DecoupledActorCriticVLM_COT(
        vlm_name=args.vlm_name,
        max_new_tokens=args.max_new_tokens,
        use_lora=True,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    
    params = agent.get_trainable_params()
    if accelerator.is_main_process:
        print("============ Agent =============")
        print(agent)
        print("\n--- Trainable Parameters ---")
        print_trainable_parameters(agent.vlm.model)
        print("-" * 50)
        print(f"Total trainable parameters: {sum(p.numel() for p in params)}")
        print("-" * 50)
        
    optimizer = optim.AdamW(params, lr=args.learning_rate, betas=(0.85, 0.9), weight_decay=args.weight_decay)
    agent, optimizer = accelerator.prepare(agent, optimizer)
    
    if args.checkpoint_dir != "":
        print(f"Loading checkpoint from {args.checkpoint_dir}")
        accelerator.load_state(args.checkpoint_dir)
        print(f"Checkpoint loaded from {args.checkpoint_dir}")
    accelerator.wait_for_everyone()
    
    # --- Storage Tensors ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    prompt_lens = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, args.max_seq_len), dtype=torch.long, device=device)
    
    # initialize full_input_ids with the pad token id
    pad_token_id = agent.vlm.processor.tokenizer.pad_token_id
    if accelerator.is_main_process: print(f"Pad token id: {pad_token_id}")
    full_input_ids = torch.full(
        (args.num_steps, args.num_envs, args.max_seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs, args.max_seq_len), device=device)

    # --- Start training ---
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    first_model_save = True
    if accelerator.is_main_process:
        current_episode_frames = []
        last_completed_episode_frames = []
    
    for iteration in range(1, args.num_iterations + 1):
        generation_lengths = []
        seq_len_errors = []
        rollout_time_start = time.time()
        
        # anneal lr
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase: Each process collects its own data ---
        for step in range(0, args.num_steps):
            if accelerator.is_main_process:
                global_step += args.num_envs * accelerator.num_processes
                current_episode_frames.append(next_obs[0].cpu().numpy().astype(np.uint8))

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # sample actions from actor
                logprob, f_ids, p_len, generated_texts = agent.get_action(
                    obs=next_obs, 
                    text_prompts=[prompt_text_actor] * args.num_envs,
                    action_ids=None,
                    prompt_lens=None,
                )
                # parse model outputs to environment action space
                action = torch.tensor(
                    [parse_action_cot(text, envs.single_action_space, action_map) for text in generated_texts],
                    device=device
                )
                # get value estimate for the current obs
                value = agent.get_value(
                    obs=next_obs,
                    prompt_text=[prompt_text_critic] * args.num_envs
                )
                values[step] = value.flatten()

            # The generated full_ids (f_ids) might be shorter or longer than max_seq_len
            seq_len = f_ids.shape[1]
            p_len = p_len.unsqueeze(1)

            # We need to create padded/truncated versions of everything to fit our storage tensors
            padded_ids = torch.full(
                (args.num_envs, args.max_seq_len),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device
            )
            padded_logprob = torch.zeros(
                (args.num_envs, args.max_seq_len),
                dtype=torch.float32,
                device=device
            )

            # Truncate or pad the generated sequence and its logprobs
            copy_len_ids = min(seq_len, args.max_seq_len)
            padded_ids[:, :copy_len_ids] = f_ids[:, :copy_len_ids]
            logprob_len = logprob.shape[1] # This will be seq_len - 1
            copy_len_logprob = min(logprob_len, args.max_seq_len - 1)
            padded_logprob[:, 1 : 1 + copy_len_logprob] = logprob[:, :copy_len_logprob]

            # Now, store the consistently shaped tensors
            full_input_ids[step] = padded_ids
            logprobs[step] = padded_logprob
            prompt_lens[step] = p_len.squeeze()

            # Recompute mask for the correctly padded/truncated sequence
            indices = torch.arange(args.max_seq_len, device=device).unsqueeze(0)
            action_token_mask = indices >= p_len
            pad_mask = (padded_ids != pad_token_id)
            action_masks[step] = (action_token_mask & pad_mask).long()
            
            generation_len = seq_len - p_len[0].cpu().item()
            seq_len_error = 1 if seq_len >= args.max_seq_len else 0
            
            generation_lengths.append(generation_len)
            seq_len_errors.append(seq_len_error)
            
            # step the environment
            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            print(f"[Process {accelerator.process_index}] Step {step+1}/{args.num_steps}")
            
            rewards[step] = torch.tensor(reward).to(device).view(-1) * 0.01
            next_done = np.logical_or(term, trunc)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if accelerator.is_main_process:
                if next_done[0]:
                    current_episode_frames.append(next_obs[0].cpu().numpy().astype(np.uint8))
                    last_completed_episode_frames = list(current_episode_frames)
                    current_episode_frames = []

            # --- Logging ---
            if accelerator.is_main_process:
                if iteration % args.log_every == 0 and step == 0:
                    log_entry_text = (
                        f"## Iteration {iteration}, Step {step} (Global Step: {global_step})\n\n"
                        f"**Prompt Actor:**\n```\n{prompt_text_actor}\n```\n\n"
                        f"**Prompt Critic:**\n```\n{prompt_text_critic}\n```\n\n"
                        f"**VLM Output (Env 0, Step 0):**\n```json\n{generated_texts[0]}\n```\n\n"
                    )

                    if last_completed_episode_frames:
                        try:
                            pil_images_debug = [Image.fromarray(frame) for frame in last_completed_episode_frames]
                            gif_path = os.path.join(log_path, f"iter_{iteration}_episode.gif")
                            
                            pil_images_debug[0].save(
                                gif_path,
                                save_all=True,
                                append_images=pil_images_debug[1:],
                                duration=150,
                                loop=0
                            )
                            
                            log_entry_text += (
                                f"**Last Completed Episode (Env 0):**\n"
                                f"![Episode GIF]({os.path.basename(gif_path)})\n\n"
                            )

                            if args.track:
                                video_array = np.array(last_completed_episode_frames)
                                video_tensor = torch.tensor(video_array).permute(0, 3, 1, 2).unsqueeze(0)
                                writer.add_video("debug/episode_video", video_tensor, global_step, fps=10)
                            
                            last_completed_episode_frames = []

                        except Exception as e:
                            print(f"Warning: Failed to save episode GIF. Error: {e}")
                            log_entry_text += f"**(Failed to save GIF: {e})**\n\n"
                    
                    log_entry_text += "---\n\n"

                    interaction_log_file.write(log_entry_text)
                    interaction_log_file.flush()

                    if args.track:
                        writer.add_text("debug/vlm_output", str(generated_texts[0]), global_step)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # Free up GPU memory after each step
            del reward, term, trunc, infos, logprob, f_ids, p_len, action, value, generated_texts, seq_len, padded_ids, padded_logprob, indices, action_token_mask, pad_mask
            gc_cuda_cleanup()
        
        rollout_time_completed = time.time() - rollout_time_start

        # --- bootstrap last value to do GAE ---
        with torch.no_grad():
            next_value = agent.get_value(obs=next_obs, prompt_text=[prompt_text_critic] * args.num_envs).flatten()

        # all processes will have identical data
        gathered_rewards = accelerator.gather(rewards) # (num_steps * num_processes, num_envs)
        gathered_values = accelerator.gather(values) # (num_steps * num_processes, num_envs)
        gathered_dones = accelerator.gather(dones) # (num_steps * num_processes, num_envs)
        gathered_next_value = accelerator.gather(next_value) # (num_envs * num_processes)
        gathered_next_done = accelerator.gather(next_done) # (num_envs * num_processes)

        num_total_envs = args.num_envs * accelerator.num_processes
        gathered_rewards = gathered_rewards.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # (num_steps, num_envs * num_processes)
        gathered_values = gathered_values.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # (num_steps, num_envs * num_processes)
        gathered_dones = gathered_dones.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs) # (num_steps, num_envs * num_processes)
        gathered_next_value = gathered_next_value.view(num_total_envs)
        gathered_next_done = gathered_next_done.view(num_total_envs)

        # all processes do GAE on the same data
        with torch.no_grad():
            advantages = torch.zeros_like(gathered_rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - gathered_next_done.float()
                    nextvalues = gathered_next_value
                else:
                    nextnonterminal = 1.0 - gathered_dones[t + 1].float()
                    nextvalues = gathered_values[t + 1]
                delta = gathered_rewards[t] + args.gamma * nextvalues * nextnonterminal - gathered_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + gathered_values

        # all processes will have identical data
        gathered_obs = accelerator.gather(obs)
        gathered_logprobs = accelerator.gather(logprobs)
        gathered_input_ids = accelerator.gather(full_input_ids)
        gathered_prompt_lens = accelerator.gather(prompt_lens)
        gathered_action_masks = accelerator.gather(action_masks)

        # all processes will have identical data
        b_action_masks = gathered_action_masks.view(
            accelerator.num_processes, args.num_steps, args.num_envs, args.max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, args.max_seq_len)

        b_logprobs = gathered_logprobs.view(
            accelerator.num_processes, args.num_steps, args.num_envs, args.max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, args.max_seq_len)

        b_input_ids = gathered_input_ids.view(
            accelerator.num_processes, args.num_steps, args.num_envs, args.max_seq_len
        ).permute(1, 0, 2, 3).reshape(-1, args.max_seq_len)

        b_prompt_lens = gathered_prompt_lens.view(
            accelerator.num_processes, args.num_steps, args.num_envs
        ).permute(1, 0, 2).reshape(-1)
        
        b_obs = gathered_obs.view(
            accelerator.num_processes, args.num_steps, args.num_envs, *envs.single_observation_space.shape
        ).permute(1, 0, 2, 3, 4, 5).reshape(-1, *envs.single_observation_space.shape) # [num_envs * num_steps * num_processes, *envs.single_observation_space.shape]

        # b_advantages = advantages.view(
        #     accelerator.num_processes, args.num_steps, args.num_envs
        # ).permute(1, 0, 2).reshape(-1)
        
        # b_returns = returns.view(
        #     accelerator.num_processes, args.num_steps, args.num_envs
        # ).permute(1, 0, 2).reshape(-1)
        
        # b_values = gathered_values.view(
        #     accelerator.num_processes, args.num_steps, args.num_envs
        # ).permute(1, 0, 2).reshape(-1)
        
        b_advantages = advantages.reshape(-1)    
        b_returns = returns.reshape(-1)
        b_values = gathered_values.reshape(-1)
        
        # --- Training ---
        b_inds = np.arange(args.total_batch_size)
        is_critic_warmup = iteration <= args.critic_warmup_iterations
        
        if not is_critic_warmup and first_model_save and args.checkpoint_dir == "":
            print(f"Saving checkpoint at iteration {iteration}")
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f"runs/{run_name}/post-critic-warmup-{iteration}")
            print(f"Checkpoint saved to runs/{run_name}/post-critic-warmup-{iteration}")
            first_model_save = False
        
        if accelerator.is_main_process:
            print("Training...")
        
        true_update_epochs = args.warmup_epochs if is_critic_warmup else args.update_epochs
        epoch_iter = range(true_update_epochs)
        if accelerator.is_main_process:
            epoch_iter = tqdm(epoch_iter, desc="Epochs")

        cliprange_low = args.clip_coef_lower
        cliprange_high = args.clip_coef_upper
        dual_clip_c = args.dual_clip_c
        ratios_1st_epoch_1st_minibatch = []
        logprobs_barplot_logged = False

        learning_time_start = time.time()
        all_values_stats = []
        all_advantages_stats = []
        all_returns_stats = []
        all_newlogprobs_stats = []
        all_oldlogprobs_stats = []
        all_logratios_stats = []
        all_ratios_stats = []
        epoch_clipfracs_upper = []
        epoch_clipfracs_lower = []
        epoch_approx_kls = []
        epoch_old_approx_kls = []
        np.random.shuffle(b_inds)

        # ----------- UNIFIED ACTOR-CRITIC (policy & value) UPDATE LOOP -----------
        
        # Normalize advantages at batch level before epoch loop
        if args.norm_adv:
            adv_mean = b_advantages.mean()
            adv_std = b_advantages.std()
            b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

        for epoch in epoch_iter:
            minibatch_iter = range(0, args.total_batch_size, args.minibatch_size)
            if accelerator.is_main_process:
                minibatch_iter = tqdm(
                    minibatch_iter,
                    desc=f"Epoch {epoch+1}/{args.update_epochs} - {'Critic Warmup' if is_critic_warmup else 'Actor-Critic'}",
                    leave=False,
                    dynamic_ncols=True,
                )

            for mb_idx, start in enumerate(minibatch_iter):
                if accelerator.is_main_process and hasattr(minibatch_iter, "set_postfix"):
                    minibatch_iter.set_postfix({"minibatch": mb_idx})

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                process_mb_inds = mb_inds[accelerator.process_index::accelerator.num_processes]
                with accelerator.accumulate(agent):
                    mb_obs = b_obs[process_mb_inds].to(device)
                    mb_input_ids = b_input_ids[process_mb_inds].to(device)
                    mb_prompt_lens = b_prompt_lens[process_mb_inds].to(device)
                    mb_logprobs = b_logprobs[process_mb_inds].to(device)
                    mb_returns = b_returns[process_mb_inds].to(device)
                    mb_values = b_values[process_mb_inds].to(device)
                    mb_advantages = b_advantages[process_mb_inds].to(device)
                    mb_action_masks = b_action_masks[process_mb_inds].to(device)

                    def stats(x):
                        x = x.float()
                        return dict(
                            mean=float(x.mean().cpu()), std=float(x.std().cpu()),
                            min=float(x.min().cpu()), max=float(x.max().cpu()), median=float(x.median().cpu())
                        )

                    all_values_stats.append(stats(mb_values))
                    all_advantages_stats.append(stats(mb_advantages))
                    all_returns_stats.append(stats(mb_returns))

                    # Only compute and backward value loss during critic warmup
                    if is_critic_warmup:
                        # ---- CRITIC VALUE LOSS ----
                        newvalue = agent.get_value(
                            obs=mb_obs, prompt_text=[prompt_text_critic] * mb_obs.shape[0]
                        ).view(-1)

                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - mb_returns) ** 2
                            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cliprange_low, cliprange_high)
                            v_loss_clipped = (v_clipped - mb_returns) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                        accelerator.backward(v_loss)

                        # optimizer step and zero grad
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Clean up minibatch objects
                        del mb_obs, mb_input_ids, mb_prompt_lens
                        del mb_logprobs, mb_advantages, mb_returns, mb_values, mb_action_masks
                        if 'newvalue' in locals():
                            del newvalue
                        gc_cuda_cleanup()
                        continue  # Skip policy optimization during critic warmup

                    # --- ACTOR POLICY LOSS ----
                    newlogprob, entropy_tensor = agent.get_action(
                        obs=mb_obs,
                        text_prompts=[prompt_text_actor] * mb_obs.shape[0],
                        action_ids=mb_input_ids,
                        prompt_lens=mb_prompt_lens
                    )
                    
                    old_logprobs_sliced = mb_logprobs[:, 1:]
                    action_masks_sliced = mb_action_masks[:, 1:]
                    logratio = newlogprob - old_logprobs_sliced
                    
                    # Clamp logratio to avoid exploding gradients
                    if args.logratio_clamp > 0:
                        logratio = torch.clamp(logratio, -args.logratio_clamp, args.logratio_clamp)
                        
                    logratio = torch.where(action_masks_sliced.bool(), logratio, torch.zeros_like(logratio))
                    ratio = torch.exp(logratio)

                    mask = action_masks_sliced.bool()
                    mask_sum = mask.sum().item()
                    if mask_sum > 0:
                        newlogprob_masked = newlogprob[mask].detach().cpu()
                        logratio_masked = logratio[mask].detach().cpu()
                        ratio_masked = ratio[mask].detach().cpu()
                        oldlogprob_masked = old_logprobs_sliced[mask].detach().cpu() 
                        
                        all_newlogprobs_stats.append(stats(newlogprob_masked))
                        all_oldlogprobs_stats.append(stats(oldlogprob_masked))
                        all_logratios_stats.append(stats(logratio_masked))
                        all_ratios_stats.append(stats(ratio_masked))

                    if (epoch == 0 and start == 0 and not logprobs_barplot_logged and accelerator.is_main_process):
                        for mb_idx_inner in range(mb_obs.shape[0]):
                            inner_mask = action_masks_sliced[mb_idx_inner].bool()
                            if inner_mask.any():
                                ratios_debug = ratio[mb_idx_inner][inner_mask].mean().cpu().item()
                                ratios_1st_epoch_1st_minibatch.append(ratios_debug)
                                print(f"Ratios debug: {ratios_debug}")

                    # KL and clip fraction stats (for logging plots and scalars)
                    with torch.no_grad():
                        valid = action_masks_sliced.bool()
                        valid_token_count = valid.sum()
                        if valid_token_count > 0:
                            masked_logratio = logratio[valid]
                            masked_ratio = ratio[valid]
                            old_approx_kl = (-masked_logratio).mean()
                            approx_kl = ((masked_ratio - 1) - masked_logratio).mean()
                            epoch_approx_kls.append(approx_kl.item())
                            epoch_old_approx_kls.append(old_approx_kl.item())

                    # dual-clip PPO loss and fractions
                    mb_advantages_exp = mb_advantages.unsqueeze(-1) 
                    
                    pg_losses1 = -mb_advantages_exp * ratio
                    pg_losses2 = -mb_advantages_exp * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
                    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

                    upper_clipfrac_mask = (pg_losses2 > pg_losses1).float()
                    if action_masks_sliced.sum() > 0:
                        clipfrac_upper = (upper_clipfrac_mask * action_masks_sliced.float()).sum() / action_masks_sliced.float().sum()
                    else:
                        clipfrac_upper = torch.tensor(0.0, device=upper_clipfrac_mask.device)
                    epoch_clipfracs_upper.append(clipfrac_upper.item())

                    pg_losses3 = -mb_advantages_exp * dual_clip_c
                    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

                    lower_clipfrac_mask = (clip_pg_losses1 > pg_losses3) & (mb_advantages_exp < 0)
                    if action_masks_sliced.sum() > 0:
                        clipfrac_lower = (lower_clipfrac_mask.float() * action_masks_sliced.float()).sum() / action_masks_sliced.float().sum()
                    else:
                        clipfrac_lower = torch.tensor(0.0, device=upper_clipfrac_mask.device)
                    epoch_clipfracs_lower.append(clipfrac_lower.item())

                    if dual_clip_c > 1.0:
                        pg_loss_per_token = torch.where(mb_advantages_exp < 0, clip_pg_losses2, clip_pg_losses1)
                    else:
                        pg_loss_per_token = clip_pg_losses1

                    pg_loss = (pg_loss_per_token * action_masks_sliced).sum() / action_masks_sliced.sum()

                    # entropy loss 
                    entropy_loss = (entropy_tensor * action_masks_sliced).sum() / action_masks_sliced.sum()
                    
                    accelerator.backward(pg_loss - args.ent_coef * entropy_loss) 

                    # ---- CRITIC VALUE LOSS ----
                    newvalue = agent.get_value(
                        obs=mb_obs, prompt_text=[prompt_text_critic] * mb_obs.shape[0]
                    ).view(-1)

                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cliprange_low, cliprange_high)
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    # backward value loss separately
                    accelerator.backward(v_loss)

                    # optimizer step ONCE for both actor and critic, then zero grad
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clean up minibatch objects after joint step
                    del mb_obs, mb_input_ids, mb_prompt_lens
                    del mb_logprobs, mb_advantages, mb_returns, mb_values, mb_action_masks
                    if 'newlogprob' in locals():
                        del newlogprob
                    if 'entropy_tensor' in locals():
                        del entropy_tensor
                    if 'logratio' in locals():
                        del logratio
                    if 'ratio' in locals():
                        del ratio
                    if 'mask' in locals():
                        del mask
                    if 'newvalue' in locals():
                        del newvalue
                    gc_cuda_cleanup()


        if iteration % args.checkpoint_interval == 0:
            print(f"Saving checkpoint at iteration {iteration}")
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=f"runs/{run_name}/checkpoint-{iteration}")
            print(f"Checkpoint saved to runs/{run_name}/checkpoint-{iteration}")

        learning_time_completed = time.time() - learning_time_start
        # --- Logging ---
        if accelerator.is_main_process:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/global_step", global_step, global_step)
            writer.add_scalar("charts/iteration", iteration, global_step)
            writer.add_scalar("debug/generation_length", np.mean(generation_lengths), global_step)
            writer.add_scalar("debug/seq_len_errors", np.mean(seq_len_errors), global_step)
            writer.add_scalar("debug/rollout_time", rollout_time_completed, global_step)
            writer.add_scalar("debug/learning_time", learning_time_completed, global_step)

            if is_critic_warmup:
                writer.add_scalar("info/critic_warmup_phase", 1, global_step)
            else:
                writer.add_scalar("info/critic_warmup_phase", 0, global_step)

            if 'v_loss' in locals():
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)

            if not is_critic_warmup:
                if 'pg_loss' in locals():
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                if 'entropy_loss' in locals():
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

            if not is_critic_warmup:
                if ratios_1st_epoch_1st_minibatch:
                    writer.add_scalar("losses/ratios_1st_epoch_1st_minibatch", np.mean(ratios_1st_epoch_1st_minibatch), global_step)
                if epoch_approx_kls:
                    writer.add_scalar("losses/approx_kl", np.mean(epoch_approx_kls), global_step)
                if epoch_old_approx_kls:
                    writer.add_scalar("losses/old_approx_kl", np.mean(epoch_old_approx_kls), global_step)
                if epoch_clipfracs_upper:
                    writer.add_scalar("losses/clipfrac_upper", np.mean(epoch_clipfracs_upper), global_step)
                if epoch_clipfracs_lower:
                    writer.add_scalar("losses/clipfrac_lower", np.mean(epoch_clipfracs_lower), global_step)

            if len(all_newlogprobs_stats) > 0 and not is_critic_warmup:
                keys = all_newlogprobs_stats[0].keys()
                avgstats = lambda arr: {k: float(np.mean([d[k] for d in arr])) for k in keys}
                writer.add_scalar("stats/newlogprob_mean", avgstats(all_newlogprobs_stats)["mean"], global_step)
                writer.add_scalar("stats/newlogprob_std", avgstats(all_newlogprobs_stats)["std"], global_step)
                writer.add_scalar("stats/newlogprob_min", avgstats(all_newlogprobs_stats)["min"], global_step)
                writer.add_scalar("stats/newlogprob_max", avgstats(all_newlogprobs_stats)["max"], global_step)
                writer.add_scalar("stats/oldlogprob_mean", avgstats(all_oldlogprobs_stats)["mean"], global_step)
                writer.add_scalar("stats/oldlogprob_std", avgstats(all_oldlogprobs_stats)["std"], global_step)
                writer.add_scalar("stats/oldlogprob_min", avgstats(all_oldlogprobs_stats)["min"], global_step)
                writer.add_scalar("stats/oldlogprob_max", avgstats(all_oldlogprobs_stats)["max"], global_step)
                writer.add_scalar("stats/logratio_mean", avgstats(all_logratios_stats)["mean"], global_step)
                writer.add_scalar("stats/logratio_std", avgstats(all_logratios_stats)["std"], global_step)
                writer.add_scalar("stats/logratio_min", avgstats(all_logratios_stats)["min"], global_step)
                writer.add_scalar("stats/logratio_max", avgstats(all_logratios_stats)["max"], global_step)
                writer.add_scalar("stats/ratio_mean", avgstats(all_ratios_stats)["mean"], global_step)
                writer.add_scalar("stats/ratio_std", avgstats(all_ratios_stats)["std"], global_step)
                writer.add_scalar("stats/ratio_min", avgstats(all_ratios_stats)["min"], global_step)
                writer.add_scalar("stats/ratio_max", avgstats(all_ratios_stats)["max"], global_step)
            if len(all_values_stats) > 0:
                keys = all_values_stats[0].keys()
                avgstats = lambda arr: {k: float(np.mean([d[k] for d in arr])) for k in keys}
                writer.add_scalar("stats/values_mean", avgstats(all_values_stats)["mean"], global_step)
                writer.add_scalar("stats/values_std", avgstats(all_values_stats)["std"], global_step)
                writer.add_scalar("stats/values_min", avgstats(all_values_stats)["min"], global_step)
                writer.add_scalar("stats/values_max", avgstats(all_values_stats)["max"], global_step)
            if len(all_advantages_stats) > 0:
                keys = all_advantages_stats[0].keys()
                avgstats = lambda arr: {k: float(np.mean([d[k] for d in arr])) for k in keys}
                writer.add_scalar("stats/advantages_mean", avgstats(all_advantages_stats)["mean"], global_step)
                writer.add_scalar("stats/advantages_std", avgstats(all_advantages_stats)["std"], global_step)
                writer.add_scalar("stats/advantages_min", avgstats(all_advantages_stats)["min"], global_step)
                writer.add_scalar("stats/advantages_max", avgstats(all_advantages_stats)["max"], global_step)
            if len(all_returns_stats) > 0:
                keys = all_returns_stats[0].keys()
                avgstats = lambda arr: {k: float(np.mean([d[k] for d in arr])) for k in keys}
                writer.add_scalar("stats/returns_mean", avgstats(all_returns_stats)["mean"], global_step)
                writer.add_scalar("stats/returns_std", avgstats(all_returns_stats)["std"], global_step)
                writer.add_scalar("stats/returns_min", avgstats(all_returns_stats)["min"], global_step)
                writer.add_scalar("stats/returns_max", avgstats(all_returns_stats)["max"], global_step)

            sps = int(args.total_batch_size / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            if is_critic_warmup:
                print(
                    f"(CRITIC WARMUP) SPS: {sps} || value.loss : {v_loss.item() if 'v_loss' in locals() else 'n/a'}"
                )
            else:
                print(
                    f"SPS: {sps} || value.loss : {v_loss.item() if 'v_loss' in locals() else 'n/a'}, policy.loss : {pg_loss.item() if 'pg_loss' in locals() else 'n/a'}, policy.entropy : {entropy_loss.item() if 'entropy_loss' in locals() else 'n/a'}",
                    "\n"
                    f"    newlogprob: {avgstats(all_newlogprobs_stats)['mean'] if all_newlogprobs_stats else 'n/a'} | "
                    f"oldlogprob: {avgstats(all_oldlogprobs_stats)['mean'] if all_oldlogprobs_stats else 'n/a'} | "
                    f"ratio: {avgstats(all_ratios_stats)['mean'] if all_ratios_stats else 'n/a'} | "
                    f"values: {avgstats(all_values_stats)['mean'] if all_values_stats else 'n/a'} | "
                    f"advantages: {avgstats(all_advantages_stats)['mean'] if all_advantages_stats else 'n/a'} | "
                    f"returns: {avgstats(all_returns_stats)['mean'] if all_returns_stats else 'n/a'}"
                )

    envs.close()
    writer.close()
