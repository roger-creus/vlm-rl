#!/usr/bin/env python3
"""
vlm_describe_simple.py

Minimal usage:
  python vlm_describe_simple.py /path/to/image.jpg "Describe this image."

This writes a simple timestamped text log to ./vlm_logs/ by default.
"""

import os
import time
import json
from dataclasses import dataclass

from PIL import Image
import numpy as np
import torch
import tyro

# Import your VLM wrapper (same as in your project)
from src.models.model import DecoupledActorCriticVLM

@dataclass
class Args:
    image_path: str
    prompt: str
    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    use_lora: bool = False
    output_dir: str = "vlm_logs"

def load_image_as_obs(image_path: str) -> np.ndarray:
    """Load image and return an obs shaped (1, H, W, 3) uint8 (compatible with your other script)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)  # (H, W, 3), uint8
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

def safe_get_generated_texts(agent, obs_tensor: torch.Tensor, prompt: str, kwargs=None):
    """
    Robustly call agent.get_action and return a list of generated strings.
    kwargs is optionally passed to the model (kept for flexibility but empty by default).
    """
    if kwargs is None:
        kwargs = {}
    # Ensure batch dim
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(0)

    try:
        out = agent.get_action(obs=obs_tensor, text_prompts=[prompt], **kwargs)
    except TypeError:
        try:
            out = agent.get_action(obs_tensor, [prompt])
        except Exception as e:
            raise RuntimeError(f"Failed to call agent.get_action: {e}") from e

    # assume last element is generated_texts (common in your wrapper)
    if isinstance(out, (tuple, list)):
        generated = out[-1]
    else:
        generated = out

    # normalize to list[str]
    gen_list = []
    if isinstance(generated, str):
        gen_list = [generated]
    elif isinstance(generated, (list, tuple, np.ndarray)):
        for x in generated:
            if isinstance(x, bytes):
                gen_list.append(x.decode("utf-8", errors="ignore"))
            else:
                gen_list.append(str(x))
    else:
        gen_list = [str(generated)]
    return gen_list

def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image
    obs = load_image_as_obs(args.image_path)
    obs_tensor = torch.from_numpy(obs).to(device)

    # Load VLM
    agent = DecoupledActorCriticVLM(vlm_name=args.vlm_name, max_new_tokens=512, use_lora=args.use_lora)
    agent.to(device)
    agent.eval()

    # Get generated text(s)
    try:
        generated_texts = safe_get_generated_texts(agent, obs_tensor, args.prompt)
    except Exception as e:
        print("Error generating text from VLM:", e)
        return

    # Build a simple log entry and append to a text file
    ts = int(time.time())
    logfile = os.path.join(args.output_dir, "vlm_descriptions.txt")
    entry = {
        "timestamp": ts,
        "time_readable": time.ctime(ts),
        "image_path": args.image_path,
        "prompt": args.prompt,
        "vlm_name": args.vlm_name,
        "outputs": generated_texts,
    }

    # Append JSON line for easy parsing later
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print results to console for immediate feedback
    print(f"\nLogged outputs to: {logfile}")
    print("Generated texts:")
    for i, txt in enumerate(generated_texts, start=1):
        print(f" [{i}] {txt}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
