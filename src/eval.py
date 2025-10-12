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
import imageio # For saving GIFs

# Assuming these are in your project's src folder
from src.models.model import DecoupledActorCriticVLM
from src.utils.utils import parse_action
from src.utils.action_maps import action_maps

@dataclass
class Args:
    """Arguments for VLM evaluation."""
    exp_name: str = "vlm-zeroshot-eval"
    """The name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""
    env_id: str = "MsPacmanNoFrameskip-v4"
    """The id of the environment."""
    vlm_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """The name of the VLM to use (HuggingFace identifier)."""
    prompt_actor_path: str = "prompt_actor.txt"
    """Path to the text file containing the actor's prompt."""

    # --- Evaluation Arguments ---
    num_eval_episodes: int = 1
    """The number of episodes to run for evaluation."""
    max_new_tokens: int = 128
    """The maximum number of new tokens to generate for an action."""
    
    # --- Device Arguments ---
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda"
    """The device to use for evaluation."""


def evaluate():
    """Main evaluation function."""
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # --- 1. Set up logging directory ---
    eval_dir = os.path.join("evals", run_name)
    os.makedirs(eval_dir, exist_ok=True)
    print(f"✅ Logging enabled. Saving results to: {eval_dir}")

    # --- 2. Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 3. Environment and Action Setup ---
    env = gym.make(args.env_id, render_mode="rgb_array")
    
    # Load the action map and actor prompt
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
    )
    agent.to(device)
    agent.eval() # Set the model to evaluation mode
    print(f"🤖 VLM '{args.vlm_name}' loaded on device: {device}")

    # --- 5. Evaluation Loop ---
    episode_rewards = []
    episode_lengths = []
    vlm_outputs_log_path = os.path.join(eval_dir, "vlm_outputs.log")

    print(f"\n🚀 Starting evaluation for {args.num_eval_episodes} episodes...")

    for episode in tqdm(range(args.num_eval_episodes), desc="Evaluating Episodes"):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        
        # Per-episode tracking
        ep_reward = 0
        ep_len = 0
        frames = []
        
        with open(vlm_outputs_log_path, "a") as log_file:
            log_file.write(f"\n{'='*20} Episode {episode + 1} {'='*20}\n")

        while not done:
            # Capture the frame before taking a step
            frames.append(env.render())
            
            # Prepare observation tensor (add batch dimension and move to device)
            obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)

            with torch.no_grad():
                # Get the action from the VLM
                _, _, _, generated_texts = agent.get_action(
                    obs=obs_tensor, 
                    text_prompts=[prompt_text_actor]
                )
                
            vlm_output = generated_texts[0]
            action = parse_action(vlm_output, env.action_space, action_map)

            # Log VLM output for this step
            with open(vlm_outputs_log_path, "a") as log_file:
                log_file.write(f"Step {ep_len + 1}: {vlm_output}\n")

            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            print(f"Step {ep_len + 1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}")
            
            # Update episode stats
            ep_reward += reward
            ep_len += 1

            if ep_len > 5: break

        # --- Log episode results ---
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        
        # Save the episode as a GIF
        gif_path = os.path.join(eval_dir, f"episode_{episode + 1}.gif")
        imageio.mimsave(gif_path, frames, duration=0.1)
        tqdm.write(f"Episode {episode + 1}: Reward={ep_reward}, Length={ep_len}. GIF saved to {gif_path}")

    env.close()

    # --- 6. Final Logging and Summary ---
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

    # Save summary to a JSON file
    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*50)
    print("✅ Evaluation Complete!")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f} ± {std_length:.2f}")
    print(f"Full results saved to: {results_path}")
    print("="*50)


if __name__ == "__main__":
    evaluate()