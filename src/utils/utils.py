import torch
import numpy as np
import gymnasium as gym
import re

from PIL import Image

from src.utils.wrappers import NoopResetEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv, MaxAndSkipEnv, FrameSkipEnv, DiscreteActionWrapper, ScreenOnlyWrapper

def numpy_to_pil(images: np.ndarray) -> list:
    """Converts a batch of numpy array images to a list of PIL Images."""
    return [Image.fromarray(img.astype(np.uint8)) for img in images]

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=60)
        env = FrameSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        return env
    return thunk

from vizdoom import gymnasium_wrapper 
def make_vizdoom_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            "VizdoomCorridor-v0",
            render_mode="rgb_array",
            max_buttons_pressed=0,
            frame_skip=4
        )
        env = DiscreteActionWrapper(env)
        env = ScreenOnlyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def parse_action_cot(text: str, action_space: gym.spaces.Discrete, action_map: dict) -> int:
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

def parse_action(text: str, action_space: gym.spaces.Discrete, action_map: dict) -> int:
    """
    Parses the action name from the VLM's raw text output.
    Returns a random action if parsing fails.
    """
    try:
        action_str = text.strip().upper()
        if action_str in action_map:
            return action_map[action_str]
    except Exception as e:
        print(f"Error parsing action: {e}. Text: '{text}'")
    
    print(f"Warning: Could not parse action '{text}'. Returning random action.")
    return action_space.sample()

def gc_cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
def get_model_class(model_name: str):
    if "Qwen3" in model_name:
        return Qwen3VLForConditionalGeneration
    elif "Qwen2_5" in model_name:
        return Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Model {model_name} not supported")