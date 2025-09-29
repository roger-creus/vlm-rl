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
    learning_rate: float = 1e-5 # VLMs require a lower learning rate
    """the learning rate of the optimizer"""
    num_envs: int = 4 # Reduced due to VLM memory constraints
    """the number of parallel game environments"""
    num_steps: int = 8 # Reduced due to VLM memory and computational cost
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 # PPO clip coefficient
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


class Agent(nn.Module):
    def __init__(self, envs, vlm_name: str):
        super().__init__()
        # Load the VLM and its processor
        self.processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name, 
            torch_dtype=torch.bfloat16, # Use bfloat16 for memory efficiency
            trust_remote_code=True
        )
        
        # We need a separate value head. It takes the VLM's hidden state as input.
        hidden_size = self.model.config.hidden_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.critic.to(self.model.dtype)

    def get_value(self, input_ids, attention_mask):
        """
        Calculates the value based on the hidden state of the last token.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Get the hidden state of the very last token
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.critic(last_hidden_state)

    def get_action_and_value(self, images, text_prompts, action_ids=None):
        """
        Main function to generate actions or re-calculate probabilities during updates.
        """
        if action_ids is None: # --- 1. Action Generation Phase ---
            # This block runs only during data collection.
            # It requires images and prompts to generate a new action.
            
            # Prepare batched inputs for the VLM
            texts = [self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                tokenize=False,
                add_generation_prompt=True,
            ) for p in text_prompts]

            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                # Generate a new token sequence
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # Decode generated text and parse actions for the environment
            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            env_actions = torch.tensor(
                [parse_action(text, envs.single_action_space) for text in generated_texts],
                device=self.model.device
            )
            
            # The full sequence for this path is the generated sequence
            full_ids = generated_ids
            prompt_len = inputs.input_ids.shape[1]

        else: # --- 2. PPO Update (Re-calculation) Phase ---
            # This block runs during learning. It does not need images or prompts.
            # It uses the provided `action_ids` (the full stored sequences).
            full_ids = action_ids
            # We need to determine the prompt length to correctly slice for the action part.
            # NOTE: This assumes the prompt part of the sequence does not contain pad tokens,
            # which is generally a safe assumption.
            # We find the first pad token; everything before it is the prompt + action.
            # This is a bit of a hack. A more robust way would be to store prompt_len.
            # But for now, let's assume we can't easily get the original prompt length.
            # The original calculation is better, but requires storing prompt_len.
            # For now, let's stick to the original calculation and assume it's available.
            # This part of the code needs to be fixed. The prompt_len is not available here.

        # --- Log Probability Calculation and Value Estimation (Common to both phases) ---
        # This part of your code has a hidden bug: it needs `prompt_len`, which is ONLY
        # calculated in the `if` block. Let's fix this by calculating the log-probability
        # for the *entire sequence* and letting the prompt portion cancel out in the PPO ratio.
        # This is a much more robust approach.

        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
        
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        )
        logits = outputs.logits

        # --- CORRECTED LOGPROB CALCULATION ---
        # Calculate logprobs for the NEXT token prediction across the whole sequence.
        # This avoids the need to know the exact `prompt_len` during the update phase.
        
        # Shift logits to align with labels for next-token prediction
        # Logits at position `i` are used to predict token at position `i+1`
        shifted_logits = logits[:, :-1, :]
        # The labels are the original tokens, shifted left by one
        shifted_labels = full_ids[:, 1:]
        
        log_probs_all = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        log_probs = torch.gather(log_probs_all, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)

        # We need a mask for the shifted labels to ignore padding
        shifted_mask = (shifted_labels != self.processor.tokenizer.pad_token_id).float()
        log_probs = log_probs * shifted_mask
        summed_log_probs = log_probs.sum(dim=1)
        
        # Value is calculated from the hidden state of the LAST non-padded token.
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        value = self.critic(last_hidden_state)
        
        # Entropy is calculated over the logits of the generated part
        entropy = Categorical(logits=shifted_logits).entropy() * shifted_mask
        summed_entropy = entropy.sum(dim=1)

        if action_ids is None:
            # During generation, we return the parsed env_action and the full sequence
            return env_actions, summed_log_probs, summed_entropy, value, full_ids, full_attention_mask
        else:
            # During update, we just return the re-calculated values
            return summed_log_probs, summed_entropy, value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

# ==================================================================================================
# ## 3. Main Training Loop
# The loop structure is similar, but data storage and interaction are adapted for the VLM.
# We now store generated token IDs and attention masks for the PPO update.
# We also add the KL-divergence penalty to the reward.
# ==================================================================================================

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        gray_scale=False,
        stack_num=1,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    with open(args.prompt_path, "r") as f:
        prompt_text = f.read()

    # Agent setup
    agent = Agent(envs, args.vlm_name).to(device)
    # Referece model for KL penalty, as suggested in the paper to prevent policy drift
    #ref_agent = Agent(envs, args.vlm_name).to(device)
    #ref_agent.load_state_dict(agent.state_dict()) # Ensure they start identical
    #ref_agent.eval() # Reference model is not trained
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # We store raw observations, but for the VLM update we will need the full generated token sequences
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device) # Store integer actions for env
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Storage for VLM inputs, which are needed during the update phase
    max_seq_len = args.max_new_tokens + 2048 # Heuristic for prompt+generated tokens
    full_input_ids = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long).to(device)
    full_attention_masks = torch.zeros((args.num_steps, args.num_envs, max_seq_len), dtype=torch.long).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    avg_returns = deque(maxlen=20)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase ---
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Convert numpy observations to PIL Images
            pil_images = numpy_to_pil(next_obs.cpu().numpy())
            
            # Get action from VLM
            with torch.no_grad():
                action, logprob, _, value, f_ids, f_mask = agent.get_action_and_value(
                    pil_images,
                    [prompt_text] * args.num_envs,
                )
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Store the full token sequence and mask for the update step
            seq_len = f_ids.shape[1]
            full_input_ids[step, :, :seq_len] = f_ids
            full_attention_masks[step, :, :seq_len] = f_mask

            # Execute action in environment
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            print(f"Took action {action}")
            # --- KL-Divergence Penalty ---
            #with torch.no_grad():
                # Get logprobs from the reference (initial) policy
            #    ref_logprob, _, _ = ref_agent.get_action_and_value(None, None, action_ids=f_ids)
            
            # The KL divergence is estimated as the difference in log-probs
            #kl_div = logprob - ref_logprob
            # Add KL penalty to the reward. We use a negative coefficient because we subtract it from reward.
            #rewards[step] -= args.kl_coef * kl_div

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # --- GAE Calculation ---
        with torch.no_grad():
            next_pil_images = numpy_to_pil(next_obs.cpu().numpy())
            # For bootstrap value, we need to process the next observation
            next_inputs = agent.processor(
                text=[prompt_text] * args.num_envs,
                images=next_pil_images,
                return_tensors="pt",
                padding=True,
            ).to(device)
            next_value = agent.get_value(next_inputs.input_ids, next_inputs.attention_mask).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_input_ids = full_input_ids.reshape(-1, max_seq_len)
        b_attention_masks = full_attention_masks.reshape(-1, max_seq_len)

        # --- PPO Update Phase ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Re-calculate logprobs, entropy, and value for the current policy
                embed()
                newlogprob, entropy, newvalue = agent.get_action_and_value(
                    None, None, action_ids=b_input_ids[mb_inds]
                )
                newvalue = newvalue.view(-1)
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()