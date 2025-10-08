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
        env = FrameSkipEnv(env, skip=32)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk

def parse_action(text: str, action_space: gym.spaces.Discrete, action_map: dict) -> int:
    """
    Parses the 'action' field from the VLM's structured output.
    Returns a random action if parsing fails.
    """
    try:
        # Using regex to find the action string, e.g., "action": "ACTION_NAME"
        match = re.search(r'"action":\s*"([^"]+)"', text)
        if match:
            action_str = match.group(1).strip()
            if action_str.upper() in action_map:
                return action_map[action_str.upper()]
    except Exception as e:
        print(f"Error parsing action: {e}. Text: {text}")
    return action_space.sample()