"""Inv-8 one-shot: patch-coverage + token-count probe for a backbone.

Usage::

    python -m scripts.probe_backbone --backbone Qwen/Qwen3-VL-2B-Instruct \
        --min-pixels 76800 --max-pixels 76800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cleanrl_vlm.models.base_vlm import BaseVLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--min-pixels", type=int, default=76800)
    ap.add_argument("--max-pixels", type=int, default=76800)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()

    vlm = BaseVLM(args.backbone, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    # Synthesize a 4-quadrant color image.
    img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img[: args.height // 2, : args.width // 2] = [255, 0, 0]
    img[: args.height // 2, args.width // 2 :] = [0, 255, 0]
    img[args.height // 2 :, : args.width // 2] = [0, 0, 255]
    img[args.height // 2 :, args.width // 2 :] = [255, 255, 0]

    obs = torch.as_tensor(img[np.newaxis], dtype=torch.uint8, device=vlm.model.device)
    inputs = vlm.preprocess_obs_and_text(obs, ["Describe the image."])
    num_image_tokens = int(inputs.image_grid_thw.prod(dim=-1).sum().item()) if hasattr(inputs, "image_grid_thw") else -1
    input_len = int(inputs.input_ids.shape[1])

    assert num_image_tokens > 0, "Inv-8: processor emitted ZERO image tokens"

    slug = args.backbone.split("/")[-1].lower()
    out = Path(f"docs/backbone_probes/{slug}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Backbone probe — {args.backbone}\n\n"
        f"- min_pixels: {args.min_pixels}\n"
        f"- max_pixels: {args.max_pixels}\n"
        f"- test image: {args.width}x{args.height} 4-color quadrants\n"
        f"- image token count: {num_image_tokens}\n"
        f"- total input_ids length: {input_len}\n"
        f"- Inv-8 patch-coverage: {'PASS' if num_image_tokens > 0 else 'FAIL'}\n"
    )
    print(f"wrote {out} (image_tokens={num_image_tokens}, input_len={input_len})")


if __name__ == "__main__":
    main()
