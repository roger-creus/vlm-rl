"""Inv-15 — ground-truth vision probe.

Usage::

    python -m scripts.probe_vision --env-id VizdoomBasic-v1 --backbone Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class ProbeFrame:
    step: int
    label_present: bool
    vlm_answer: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--env-config", default="configs/envs/VizdoomBasic-v1.yaml")
    args = ap.parse_args()

    from cleanrl_vlm.envs.registry import make_env
    from cleanrl_vlm.envs.vizdoom.action_tables import action_tables
    from cleanrl_vlm.models.base_vlm import BaseVLM
    from cleanrl_vlm.prompts.builder import PromptBuilder

    env_cfg = yaml.safe_load(Path(args.env_config).read_text())
    min_px = int(env_cfg.get("processor_min_pixels", 262144))
    max_px = int(env_cfg.get("processor_max_pixels", 1310720))

    env = make_env(args.env_id, env_cfg, seed=0, idx=0, run_name="probe_vision")()
    action_names = action_tables[args.env_id]
    pb = PromptBuilder(args.env_id, action_names)
    probe_text = (Path(pb.templates_root) / PromptBuilder._env_id_to_slug(args.env_id) / "vision_probe.txt").read_text()

    vlm = BaseVLM(args.backbone, min_pixels=min_px, max_pixels=max_px)
    obs, _ = env.reset(seed=0)
    frames: list[ProbeFrame] = []
    for step in range(args.num_frames):
        obs_t = torch.as_tensor(obs[np.newaxis], dtype=torch.uint8, device=vlm.model.device)
        inputs = vlm.preprocess_obs_and_text(obs_t, [probe_text])
        ids = vlm.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        txt = vlm.processor.batch_decode(ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
        # Ground truth labels can be added per environment; for now the report
        # records the VLM answer for qualitative inspection.
        frames.append(ProbeFrame(step=step, label_present=True, vlm_answer=txt))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()

    slug = args.backbone.split("/")[-1].lower()
    out = Path(f"reports/vision_probes/{args.env_id}_{slug}/report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# Vision probe — {args.env_id} / {args.backbone}",
        "",
        "| Step | VLM answer |",
        "|---|---|",
    ]
    for f in frames:
        body.append(f"| {f.step} | {f.vlm_answer[:200].replace(chr(10), ' ')} |")
    out.write_text("\n".join(body))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
