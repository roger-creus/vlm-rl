"""Checkpoint save/load for VLM actor-critic trainers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def _atomic_rename(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def save_vlm_actor_critic_checkpoint(
    path: str | Path,
    algo_slug: str,
    ac_model,
    optimizer: torch.optim.Optimizer,
    rng_state: dict[str, Any],
    step: int,
    wandb_run_id: str | None,
    manifest: dict[str, Any],
) -> Path:
    """Save an atomic actor+critic checkpoint (reviewer m6).

    Layout::

        <path>/model/lora_adapters/{actor,critic}/
        <path>/model/critic_head.pt
        <path>/optimizer/optimizer.pt
        <path>/training/{step.json, rng.pt}
        <path>/logging/wandb_run_id.txt
        <path>/manifest.json
        <path>/INTEGRITY_HASHES.txt
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    model_dir = tmp / "model"
    (model_dir / "lora_adapters").mkdir(parents=True)
    ac_model.vlm.model.save_pretrained(model_dir / "lora_adapters" / "actor", selected_adapters=["actor"])
    ac_model.vlm.model.save_pretrained(model_dir / "lora_adapters" / "critic", selected_adapters=["critic"])
    torch.save(ac_model.critic_head.state_dict(), model_dir / "critic_head.pt")

    opt_dir = tmp / "optimizer"
    opt_dir.mkdir()
    torch.save(optimizer.state_dict(), opt_dir / "optimizer.pt")

    train_dir = tmp / "training"
    train_dir.mkdir()
    (train_dir / "step.json").write_text(json.dumps({"step": step, "algo_slug": algo_slug}, indent=2))
    torch.save(rng_state, train_dir / "rng.pt")

    log_dir = tmp / "logging"
    log_dir.mkdir()
    if wandb_run_id:
        (log_dir / "wandb_run_id.txt").write_text(wandb_run_id)

    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

    hashes = []
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes.append(f"{p.relative_to(tmp)}  {h}")
    (tmp / "INTEGRITY_HASHES.txt").write_text("\n".join(hashes))

    _atomic_rename(tmp, path)
    return path


def load_vlm_actor_critic_checkpoint(
    path: str | Path,
    ac_model,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    path = Path(path)
    state: dict[str, Any] = {}
    state["step"] = json.loads((path / "training" / "step.json").read_text())
    state["rng"] = torch.load(path / "training" / "rng.pt")
    ac_model.critic_head.load_state_dict(torch.load(path / "model" / "critic_head.pt"))
    optimizer.load_state_dict(torch.load(path / "optimizer" / "optimizer.pt"))
    # LoRA adapter loading is handled by the trainer via
    # ac_model.vlm.model.load_adapter(...).
    return state
