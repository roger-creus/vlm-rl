"""InvariantMonitor scaffold + per-Inv check functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class InvariantResult:
    name: str
    status: str  # "green" | "red" | "skipped"
    detail: str = ""


@dataclass
class InvariantMonitor:
    checks: dict[str, Callable[..., InvariantResult]] = field(default_factory=dict)
    sample_every: int = 10

    def register(self, name: str, fn: Callable[..., InvariantResult]) -> None:
        self.checks[name] = fn

    def maybe_run(self, step: int, ctx: dict[str, Any]) -> dict[str, InvariantResult]:
        if step % self.sample_every != 0:
            return {}
        results: dict[str, InvariantResult] = {}
        for name, fn in self.checks.items():
            try:
                results[name] = fn(ctx)
            except Exception as e:  # surface as red, never crash training
                results[name] = InvariantResult(name=name, status="red", detail=str(e))
                log.exception("Invariant %s raised", name)
        return results


# --- per-Inv check functions --------------------------------------------------


def check_inv_01_lora_trainability(ctx: dict[str, Any]) -> InvariantResult:
    ac = ctx["ac_model"]
    for n, p in ac.vlm.model.named_parameters():
        expected = "lora_" in n
        if p.requires_grad != expected:
            return InvariantResult("inv_01", "red", f"{n} requires_grad={p.requires_grad}")
    return InvariantResult("inv_01", "green")


def check_inv_05_grad_norm(ctx: dict[str, Any]) -> InvariantResult:
    """Cross-check ``clip_grad_norm_`` vs manual ``sqrt(sum(sum(g**2)))``."""
    import math

    import torch

    params = [p for p in ctx["params"] if p.grad is not None]
    if not params:
        return InvariantResult("inv_05", "skipped", "no grads")
    manual = math.sqrt(sum(float(p.grad.pow(2).sum().item()) for p in params))
    clip_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1e30))
    if abs(manual - clip_norm) > max(1e-3, 1e-3 * manual):
        return InvariantResult("inv_05", "red", f"manual={manual} clip={clip_norm}")
    return InvariantResult("inv_05", "green", f"n={clip_norm:.4f}")


INV_04_TOLERANCE: float = 1e-4
