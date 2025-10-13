import os
import random
import time
import json
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm
import imageio
import wandb
from accelerate import Accelerator
import torch.optim as optim

# Assuming these are in your project's src folder
from src.models.model import DecoupledActorCriticVLM
from src.utils.utils import parse_action, make_env
from src.utils.action_maps import action_maps

@dataclass
class Args:
    """Arguments for VLM evaluation."""
    wandb_project_name: str = "CleanRL-VLM-Eval-ZeroShot"
    """The name of the wandb project to log under."""
    seed: int = 1
    """Seed of the experiment."""
    env_id: str = "MsPacmanNoFrameskip-v4"
    """The id of the environment."""
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """The name of the VLM to use (HuggingFace identifier)."""
    prompt_actor_path: str = "prompt_actor.txt"
    """Path to the text file containing the actor's prompt."""

    # --- Evaluation Arguments ---
    num_eval_episodes: int = 16
    """The total number of episodes to run for evaluation across all processes."""
    num_eval_envs: int = 4
    """The number of parallel environments to use *per process*."""
    max_new_tokens: int = 128
    """The maximum number of new tokens to generate for an action."""
    debug_max_steps: int = 0
    """If > 0, stop after this many steps for debugging. Default is 0 (disabled)."""

    # --- Device Arguments ---
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

def evaluate_distributed():
    args = tyro.cli(Args)
    
    # --- 1. Initialize Accelerator ---
    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.num_eval_envs
        
    device = accelerator.device
    
    model_name = args.vlm_name.split("/")[-1]
    run_name = f"{model_name}_{args.env_id}_{args.seed}_{int(time.time())}"

    # --- 2. Main Process Logging Setup ---
    if accelerator.is_main_process:
        eval_dir = os.path.join("evals", run_name)
        os.makedirs(eval_dir, exist_ok=True)
        print(f"✅ Logging enabled. Saving results to: {eval_dir}")
        
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            dir=eval_dir,
            reinit=True,
        )
        print(f"🔗 wandb logging enabled for run: {run_name}")
    else:
        eval_dir = None # Only main process knows the eval dir

    # --- 3. Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    # --- 4. Environments and Prompts ---
    # Each process runs its own set of vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i + accelerator.process_index * args.num_eval_envs, False) for i in range(args.num_eval_envs)]
    )
    
    try:
        action_map = action_maps[args.env_id]
        if accelerator.is_main_process: print(f"Loaded action map for {args.env_id}")
    except KeyError:
        if accelerator.is_main_process: print(f"⚠️ Warning: No action map found for {args.env_id}.")
        action_map = {}
        
    with open(args.prompt_actor_path, "r") as f:
        prompt_text_actor = f.read()
    if accelerator.is_main_process: print("Actor prompt loaded successfully.")

    # --- 5. Load and Prepare the VLM Agent ---
    agent = DecoupledActorCriticVLM(vlm_name=args.vlm_name, max_new_tokens=args.max_new_tokens, use_lora=False)
    agent.eval()
    dummy_optimizer = optim.Adam(agent.parameters(), lr=0.0001)
    
    # Prepare the model with Accelerate
    agent, dummy_optimizer = accelerator.prepare(agent, dummy_optimizer)
    
    if accelerator.is_main_process:
        print(f"🤖 VLM '{model_name}' prepared on {accelerator.num_processes} processes.")

    # --- 6. Distributed Evaluation Loop ---
    # Determine the number of episodes this process should handle
    episodes_per_process = args.num_eval_episodes // accelerator.num_processes
    remainder = args.num_eval_episodes % accelerator.num_processes
    if accelerator.process_index < remainder:
        episodes_to_run = episodes_per_process + 1
    else:
        episodes_to_run = episodes_per_process

    local_ep_rewards = []
    local_ep_lengths = []
    current_frames = [[] for _ in range(args.num_eval_envs)]
    
    # Each process has its own log file to avoid conflicts
    vlm_outputs_log_path = os.path.join(eval_dir, f"vlm_outputs_proc_{accelerator.process_index}.log") if accelerator.is_main_process else "vlm_outputs.log"
    with open(vlm_outputs_log_path, "w") as f:
        f.write(f"Process {accelerator.process_index} starting evaluation for {episodes_to_run} episodes...\n")

    if accelerator.is_main_process:
        print(f"\n🚀 Starting evaluation for {args.num_eval_episodes} total episodes across {accelerator.num_processes} GPUs...")
    
    obs, _ = envs.reset(seed=args.seed + accelerator.process_index)
    
    pbar = tqdm(total=episodes_to_run, desc=f"Proc {accelerator.process_index} Episodes", position=accelerator.process_index)
    
    step_count = 0
    completed_episodes_local = 0
    
    while completed_episodes_local < episodes_to_run:
        rendered_frames = envs.call("render")
        for i in range(args.num_eval_envs):
            current_frames[i].append(rendered_frames[i])
        
        obs_tensor = torch.from_numpy(obs).to(device)

        with torch.no_grad():
            _, _, _, generated_texts = agent.module.get_action(
                obs=obs_tensor, 
                text_prompts=[prompt_text_actor] * args.num_eval_envs
            )
            
        actions = [parse_action(text, envs.single_action_space, action_map) for text in generated_texts]

        with open(vlm_outputs_log_path, "a") as log_file:
            log_file.write(f"\n--- Step (Local Episodes: {completed_episodes_local}) ---\n")
            for i, text in enumerate(generated_texts):
                log_file.write(f"  Env {i}: {text.strip()}\n")

        obs, _, _, _, infos = envs.step(actions)
        
        if args.debug_max_steps > 0:
            step_count += 1
            if step_count >= args.debug_max_steps:
                if accelerator.is_main_process: print("Reached debug_max_steps. Stopping.")
                break

        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info and "episode" in info and completed_episodes_local < episodes_to_run:
                    ep_rew, ep_len = info["episode"]["r"], info["episode"]["l"]
                    local_ep_rewards.append(ep_rew)
                    local_ep_lengths.append(ep_len)
                    
                    if accelerator.is_main_process:
                        gif_path = os.path.join(eval_dir, f"episode_{wandb.run.step + 1}_proc_{accelerator.process_index}.gif")
                        imageio.mimsave(gif_path, current_frames[i], duration=0.1)
                        wandb.log({"episode_video": wandb.Video(gif_path, fps=10)})
                    
                    current_frames[i] = []
                    completed_episodes_local += 1
                    pbar.update(1)
                    
                    if accelerator.is_main_process:
                        wandb.log({"episode_reward": ep_rew, "episode_length": ep_len})

    pbar.close()
    envs.close()
    accelerator.wait_for_everyone()

    # --- 7. Gather Results and Final Logging (Main Process Only) ---
    if accelerator.is_main_process:
        # Convert local lists to tensors for gathering
        local_rewards_tensor = torch.tensor(local_ep_rewards, device=device)
        local_lengths_tensor = torch.tensor(local_ep_lengths, device=device)

        # Gather tensors from all processes
        gathered_rewards_list = accelerator.gather(local_rewards_tensor)
        gathered_lengths_list = accelerator.gather(local_lengths_tensor)
        
        # The gathered list might have tensors of different lengths. We need to flatten them.
        all_rewards = [item.item() for tensor in gathered_rewards_list for item in tensor]
        all_lengths = [item.item() for tensor in gathered_lengths_list for item in tensor]

        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        avg_length = np.mean(all_lengths)
        std_length = np.std(all_lengths)

        summary = {
            "args": vars(args),
            "results": {
                "mean_reward": avg_reward, "std_reward": std_reward,
                "mean_episode_length": avg_length, "std_episode_length": std_length,
            },
            "per_episode_rewards": all_rewards, "per_episode_lengths": all_lengths,
        }

        results_path = os.path.join(eval_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=4)

        wandb.log({
            "mean_reward": avg_reward, "std_reward": std_reward,
            "mean_episode_length": avg_length, "std_episode_length": std_length,
        })
        wandb.save(results_path)
        wandb.finish()

        print("\n" + "="*50)
        print("✅ Distributed Evaluation Complete!")
        print(f"Total episodes evaluated: {len(all_rewards)}")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")
        print(f"Full results saved to: {results_path}")
        print("="*50)

if __name__ == "__main__":
    evaluate_distributed()