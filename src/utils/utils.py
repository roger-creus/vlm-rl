import torch
import numpy as np
import gymnasium as gym
import re

from PIL import Image

from src.utils.wrappers import NoopResetEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv, MaxAndSkipEnv, FrameSkipEnv

def numpy_to_pil(images: np.ndarray) -> list:
    """Converts a batch of numpy array images to a list of PIL Images."""
    return [Image.fromarray(img.astype(np.uint8)) for img in images]

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = FrameSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        return env
    return thunk

def parse_action(text: str, action_space: gym.spaces.Discrete, action_map: dict) -> int:
    """
    Parses the 'ACTION' from the VLM's multi-line text output.
    Returns a random action if parsing fails.
    """
    try:
        parts = text.split("ACTION:")
        if len(parts) > 1:
            action_str = parts[1].strip().upper()
            if action_str in action_map:
                return action_map[action_str]
    except Exception as e:
        print(f"Error parsing action: {e}. Text: '{text}'")
    
    return action_space.sample()

def gc_cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()