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
    vlm_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    """The name of the VLM to use (HuggingFace identifier)."""
    prompt_actor_path: str = "prompt_actor.txt"
    """Path to the text file containing the actor's prompt."""

    # --- Evaluation Arguments ---
    num_eval_episodes: int = 1
    """The total number of episodes to run for evaluation."""
    num_eval_envs: int = 4
    """The number of parallel environments to use for evaluation."""
    max_new_tokens: int = 128
    """The maximum number of new tokens to generate for an action."""
    
    # --- Device Arguments ---
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda"
    """The device to use for evaluation."""

def evaluate_parallel():
    """Main parallel evaluation function."""
    args = tyro.cli(Args)
    model_name = args.vlm_name.split("/")[-1]
    run_name = f"{model_name}_{args.env_id}_{args.seed}_{int(time.time())}"

    # --- 1. Set up logging directory ---
    eval_dir = os.path.join("evals", run_name)
    os.makedirs(eval_dir, exist_ok=True)
    print(f"✅ Logging enabled. Saving results to: {eval_dir}")

    # --- 1b. Initialize wandb ---
    wandb.init(
        project=args.wandb_project_name,
        name=run_name,
        config=vars(args),
        dir=eval_dir,
        reinit=True,
    )
    print(f"🔗 wandb logging enabled under project {args.wandb_project_name}, run name: {run_name}")

    # --- 2. Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, False) for i in range(args.num_eval_envs)]
    )
    
    try:
        action_map = action_maps[args.env_id]
        print(f"Loaded action map for {args.env_id}")
    except KeyError:
        print(f"⚠️ Warning: No action map found for {args.env_id}. Action parsing might fail.")
        action_map = {}
        
    with open(args.prompt_actor_path, "r") as f:
        prompt_text_actor = f.read()
    print("Actor prompt loaded successfully.")

    # --- 4. Load the VLM Agent ---
    agent = DecoupledActorCriticVLM(
        vlm_name=args.vlm_name,
        max_new_tokens=args.max_new_tokens,
        use_lora=False,
    )
    agent.to(device)
    agent.eval() # Set the model to evaluation mode
    print(f"🤖 VLM '{model_name}' loaded on device: {device}")

    # --- 5. Evaluation Loop ---
    completed_episodes = 0
    episode_rewards = []
    episode_lengths = []
    
    # Per-environment trackers
    current_frames = [[] for _ in range(args.num_eval_envs)]
    
    vlm_outputs_log_path = os.path.join(eval_dir, "vlm_outputs.log")
    with open(vlm_outputs_log_path, "w") as f: # Clear the log file
        f.write(f"Starting parallel evaluation for {args.num_eval_episodes} total episodes...\n")

    print(f"\n🚀 Starting evaluation for {args.num_eval_episodes} episodes across {args.num_eval_envs} environments...")
    
    obs, _ = envs.reset(seed=args.seed)
    
    pbar = tqdm(total=args.num_eval_episodes, desc="Completed Episodes")
    step_count = 0
    
    while completed_episodes < args.num_eval_episodes:
        # Capture frames from all parallel environments
        rendered_frames = envs.call("render")
        for i in range(args.num_eval_envs):
            current_frames[i].append(rendered_frames[i])
        
        # Prepare observation tensor (batch is already the first dimension)
        obs_tensor = torch.from_numpy(obs).to(device)

        with torch.no_grad():
            # Get a batch of actions from the VLM
            _, _, _, generated_texts = agent.get_action(
                obs=obs_tensor, 
                text_prompts=[prompt_text_actor] * args.num_eval_envs
            )
            
        actions = [parse_action(text, envs.single_action_space, action_map) for text in generated_texts]

        # Log VLM outputs for this step
        with open(vlm_outputs_log_path, "a") as log_file:
            log_file.write(f"\n--- Step (Completed Episodes: {completed_episodes}) ---\n")
            for i, text in enumerate(generated_texts):
                log_file.write(f"  Env {i}: {text.strip()}\n")

        # Step the vectorized environments
        obs, _, _, _, infos = envs.step(actions)
        
        # Check for finished episodes in the `infos` dictionary
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info is not None and "episode" in info:
                    if completed_episodes >= args.num_eval_episodes:
                        continue # Stop logging if we've hit our target
                        
                    ep_rew = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    
                    episode_rewards.append(ep_rew)
                    episode_lengths.append(ep_len)
                    completed_episodes += 1
                    
                    # Save the GIF for the completed episode
                    gif_path = os.path.join(eval_dir, f"episode_{completed_episodes}.gif")
                    imageio.mimsave(gif_path, current_frames[i], duration=0.1)
                    
                    # Reset the frame buffer for this specific environment
                    current_frames[i] = []
                    
                    pbar.update(1)
                    pbar.set_postfix({"Last Reward": f"{ep_rew:.2f}"})

                    # --- wandb Log after each completed episode ---
                    wandb.log(
                        {
                            "episode_reward": ep_rew,
                            "episode_length": ep_len,
                            "completed_episodes": completed_episodes,
                        }
                    )

    pbar.close()
    envs.close()

    # --- 6. Final Logging and Summary ---
    # Handle the case where the loop broke early for debugging and no episodes finished
    if not episode_rewards:
        print("\nNo episodes completed. Exiting.")
        wandb.finish()
        return

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    summary = {
        "args": vars(args),
        "results": {
            "mean_reward": avg_reward,
            "std_reward": std_reward,
            "mean_episode_length": avg_length,
            "std_episode_length": std_length,
        },
        "per_episode_rewards": episode_rewards,
        "per_episode_lengths": episode_lengths,
    }

    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)

    # --- Final wandb logging ---
    wandb.log(
        {
            "mean_reward": avg_reward,
            "std_reward": std_reward,
            "mean_episode_length": avg_length,
            "std_episode_length": std_length,
            "num_eval_episodes": args.num_eval_episodes,
            "num_eval_envs": args.num_eval_envs,
        }
    )
    wandb.save(results_path)
    wandb.finish()

    print("\n" + "="*50)
    print("✅ Evaluation Complete!")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")
    print(f"Full results saved to: {results_path}")
    print("="*50)

if __name__ == "__main__":
    evaluate_parallel()