# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from PIL import Image
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from accelerate.utils import TorchDynamoPlugin
from accelerate import Accelerator 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from IPython import embed


# ==================================================================================================
# ## 1. Argument Parsing and Configuration
# We add new arguments for the VLM, prompt, and RL-specific hyperparameters from the papers.
# ==================================================================================================
@dataclass
class Args:
    exp_name: str = "vlm_ppo_finetuning"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-VLM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MsPacman-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    log_every: int = 1
    """the frequency of logging interactions to a file"""
    log_dir: str = "interaction_logs"
    """the directory to save interaction logs"""

    # VLM specific arguments
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """The model ID from Hugging Face."""
    prompt_path: str = "prompt.txt"
    """Path to a file containing the text prompt for the VLM."""
    kl_coef: float = 0.02
    """Coefficient for the KL-divergence penalty in the reward, to stabilize training."""
    cot_lambda: float = 0.5
    """Scaling factor to down-weight the CoT reasoning tokens in log-prob calculation, as per Section 4.3."""
    max_new_tokens: int = 128
    """Maximum number of new tokens for the VLM to generate."""
    enable_compile: bool = True
    "wether to compile VLM"
    
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

# ==================================================================================================
# ## 2. Environment and Agent Definition
# The core logic is now in the `Agent` class. It handles VLM loading, text generation,
# action parsing, and calculating log probabilities for text sequences.
# ==================================================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def numpy_to_pil(images: np.ndarray) -> list:
    """Converts a batch of numpy array images to a list of PIL Images."""
    # Assuming images are (N, C, H, W).
    images = images.transpose(0, 2, 3, 1) # Convert to NHWC
    return [Image.fromarray(img.astype(np.uint8)) for img in images]

# Helper function to parse the VLM's text output to an action
def parse_action(text: str, action_space: gym.spaces.Discrete) -> int:
    """
    Parses the 'action' field from the VLM's structured output.
    Returns a random action if parsing fails.
    """
    try:
        # Using regex to find the action string, e.g., "action": "ACTION_NAME"
        match = re.search(r'"action":\s*"([^"]+)"', text)
        if match:
            action_str = match.group(1).strip()
            # This is a placeholder mapping. You'll need to create a specific
            # mapping for your environment's actions.
            # E.g., for MsPacman-v5, the actions are integers 0-8.
            # We can map keywords to these integers.
            action_map = {
                "NOOP": 0, "UP": 1, "RIGHT": 2, "LEFT": 3, "DOWN": 4,
                "UPRIGHT": 5, "UPLEFT": 6, "DOWNRIGHT": 7, "DOWNLEFT": 8,
            }
            if action_str.upper() in action_map:
                return action_map[action_str.upper()]
    except Exception as e:
        print(f"Error parsing action: {e}. Text: {text}")
    # Default to a random action if parsing fails, as suggested in the paper.
    return action_space.sample()

# The RecordEpisodeStatistics wrapper remains the same
class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

class Agent(nn.Module):
    def __init__(self, envs, vlm_name: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True, min_pixels = 210 * 160 * 3, max_pixels = 210 * 160 * 3)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        hidden_size = self.model.config.hidden_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.critic.to(self.model.dtype)

    def get_last_hidden_state(self, hidden_states, attention_mask):
        """Calculates the hidden state of the last non-padding token."""
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden_state = hidden_states[batch_indices, last_token_indices, :]
        return last_hidden_state

    def get_value(self, input_ids, attention_mask):
        """Calculates value from the last non-padding token's hidden state."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        last_hidden_state = self.get_last_hidden_state(outputs.hidden_states[-1], attention_mask)
        return self.critic(last_hidden_state)

    def get_action_and_value(self, images, text_prompts, action_ids=None, prompt_lens=None):
        if action_ids is None:  # --- Generation Phase ---
            texts = [self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                tokenize=False, add_generation_prompt=True,
            ) for p in text_prompts]
            inputs = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True,
            ).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.9,
                )
            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            env_actions = torch.tensor(
                [parse_action(text, envs.single_action_space) for text in generated_texts],
                device=self.model.device
            )
            full_ids = generated_ids
            prompt_lens = torch.tensor([inputs.input_ids.shape[1]] * len(images), device=self.model.device)
        else:  # --- Update Phase ---
            full_ids = action_ids

        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
        outputs = self.model(
            input_ids=full_ids, attention_mask=full_attention_mask, output_hidden_states=True
        )
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        shifted_labels = full_ids[:, 1:]
        log_probs_all = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        log_probs = torch.gather(log_probs_all, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)

        indices = torch.arange(shifted_labels.shape[1], device=shifted_labels.device)
        action_token_mask = indices[None, :] >= (prompt_lens - 1)[:, None]
        shifted_pad_mask = (shifted_labels != self.processor.tokenizer.pad_token_id)
        final_mask = action_token_mask & shifted_pad_mask
        
        log_probs = log_probs * final_mask
        summed_log_probs = log_probs.sum(dim=1)
        
        last_hidden_state = self.get_last_hidden_state(outputs.hidden_states[-1], full_attention_mask)
        value = self.critic(last_hidden_state)
        
        entropy = Categorical(logits=shifted_logits).entropy()

        if action_ids is None:
            return env_actions, summed_log_probs, value, full_ids, full_attention_mask, prompt_lens, generated_texts
        else:
            return summed_log_probs, value, entropy, final_mask


# ==================================================================================================
# ## 4. Main Training Loop
# ==================================================================================================

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.enable_compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="reduce-overhead",      # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=True,
            dynamic=False
        )
        accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
        print("Enabling compilation!")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=1)
        
    device = accelerator.device

    # 2. Calculate batch sizes, accounting for the number of processes
    # ✨ DISTRIBUTED RL: The total batch size is now multiplied by the number of GPUs
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

    # ✨ DISTRIBUTED RL: Wait for the main process to print before continuing
    accelerator.wait_for_everyone()

    # 3. Set up DeepSpeed config programmatically
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = per_process_minibatch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # ✨ DISTRIBUTED RL: Guard all logging to run only on the main process
    if accelerator.is_main_process:
        # Create a dedicated folder for this run's interaction logs
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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ✨ DISTRIBUTED RL: Each process creates its own environments.
    # `args.num_envs` now means "environments per process".
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed + accelerator.process_index,
        stack_num=1,
        gray_scale=False,
        img_width=160,
        img_height=210,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    with open(args.prompt_path, "r") as f:
        prompt_text = f.read()

    agent = Agent(envs, args.vlm_name)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # These storage tensors now hold the data for a single process
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = torch.zeros((args.num_steps, args.num_envs))
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))
    prompt_lens = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    max_seq_len = 1024
    full_input_ids = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long)
    full_attention_masks = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase: Each process collects its own data ---
        for step in range(0, args.num_steps):
            if accelerator.is_main_process:
                global_step += args.num_envs * accelerator.num_processes
            
            obs[step] = next_obs.cpu()
            dones[step] = next_done.cpu()
            
            
            pil_images = numpy_to_pil(next_obs.cpu().numpy())
            with torch.no_grad():
                action, logprob, value, f_ids, f_mask, p_len, generated_texts = agent.get_action_and_value(
                    pil_images, [prompt_text] * args.num_envs,
                )
                values[step] = value.flatten().cpu()
                
            if accelerator.is_main_process and iteration % args.log_every == 0 and step == 0:
                log_image = pil_images[0]
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
            
            actions[step] = action.cpu()
            logprobs[step] = logprob.cpu()
            prompt_lens[step] = p_len.cpu()
            
            seq_len = min(f_ids.shape[1], max_seq_len)
            full_input_ids[step, :, :seq_len] = f_ids[:, :seq_len].cpu()
            full_attention_masks[step, :, :seq_len] = f_mask[:, :seq_len].cpu()

            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            print(f"🚀 [Process {accelerator.process_index}] Step {step+1}/{args.num_steps} - Actions: {action.cpu().numpy()}")

            rewards[step] = torch.tensor(reward)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                    
            # ✨ DISTRIBUTED RL: Only main process logs episode stats
            if accelerator.is_main_process:
                for idx, d in enumerate(next_done):
                     if d and info["lives"][idx] == 0:
                        print(f"global_step={global_step}, episodic_return={info['r'][idx]}, episodic_length={info['l'][idx]}")
                        writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # --- GAE Calculation Section ---
        # ✨ STEP 1: ALL processes must run this block in parallel ✨
        with torch.no_grad():
            next_pil_images = numpy_to_pil(next_obs.cpu().numpy())
            # Use agent.module to access the original model's processor when using accelerate
            next_inputs = agent.module.processor(
                text=[prompt_text] * args.num_envs, images=next_pil_images, return_tensors="pt", padding=True,
            ).to(device)
            # Each process computes the bootstrap value for its OWN final observation
            next_value = agent.get_value(next_inputs.input_ids, next_inputs.attention_mask).flatten()

        # ✨ STEP 2: ALL processes participate in gathering the data ✨
        gathered_rewards = accelerator.gather(rewards.to(device))
        gathered_values = accelerator.gather(values.to(device))
        gathered_dones = accelerator.gather(dones.to(device))
        gathered_next_value = accelerator.gather(next_value)
        gathered_next_done = accelerator.gather(next_done)

        # ✨ STEP 3: Reshape the gathered data ✨
        num_total_envs = args.num_envs * accelerator.num_processes
        gathered_rewards = gathered_rewards.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs)
        gathered_values = gathered_values.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs)
        gathered_dones = gathered_dones.view(accelerator.num_processes, args.num_steps, args.num_envs).permute(1, 0, 2).reshape(args.num_steps, num_total_envs)
        gathered_next_value = gathered_next_value.view(num_total_envs)
        gathered_next_done = gathered_next_done.view(num_total_envs)

        # ✨ DISTRIBUTED RL: GAE calculation now runs on ALL processes.
        # This is redundant but fast and avoids deadlocks by ensuring all processes have the data they need.
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

        # ✨ DISTRIBUTED RL: Flatten the batches on ALL processes.
        # Now every process has the identical, complete training data.
        b_logprobs = accelerator.gather(logprobs.to(device)).reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = gathered_values.reshape(-1)
        b_input_ids = accelerator.gather(full_input_ids.to(device)).reshape(-1, max_seq_len)
        b_prompt_lens = accelerator.gather(prompt_lens.to(device)).reshape(-1)

        b_inds = np.arange(args.total_batch_size)
        clipfracs = []

        if accelerator.is_main_process:
            print("Training...")

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.total_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                process_mb_inds = mb_inds[accelerator.process_index::accelerator.num_processes]
                
                with accelerator.accumulate(agent):
                    mb_input_ids = b_input_ids[process_mb_inds].to(device)
                    mb_prompt_lens = b_prompt_lens[process_mb_inds].to(device)
                    mb_logprobs = b_logprobs[process_mb_inds].to(device)
                    mb_advantages = b_advantages[process_mb_inds].to(device)
                    mb_returns = b_returns[process_mb_inds].to(device)
                    mb_values = b_values[process_mb_inds].to(device)
                    # -----------------------------------------------------------

                    newlogprob, newvalue, entropy_tensor, final_mask = agent.get_action_and_value(
                        None, None,
                        action_ids=mb_input_ids,
                        prompt_lens=mb_prompt_lens
                    )
                    newvalue = newvalue.view(-1)
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()
                    
                    #with torch.no_grad():
                    #    approx_kl = accelerator.gather( ((ratio - 1) - logratio).mean() ).mean()
                    #    clipfracs += [accelerator.gather( ((ratio - 1.0).abs() > args.clip_coef).float().mean() ).mean().item()]
                    
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    
                    entropy_loss = (entropy_tensor * final_mask).sum() / final_mask.sum()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

        # ✨ DISTRIBUTED RL: Logging should still be guarded
        if accelerator.is_main_process:
            global_step = iteration * args.total_batch_size
            
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            #writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            #writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            #writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            sps = int(args.total_batch_size / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"SPS: {sps} || value.loss : {v_loss.item()}, policy.loss : {pg_loss.item()}, policy.entropy : {entropy_loss.item()}")

    envs.close()
    writer.close()