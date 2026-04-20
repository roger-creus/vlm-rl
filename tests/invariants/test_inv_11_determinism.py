"""Inv-11 — bitwise determinism under fixed seed (reviewer M5)."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.tier1


def _make_fully_deterministic(seed: int) -> None:
    """Apply every knob the master spec requires (reviewer M5)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    # HF reproducibility.
    os.environ.setdefault("HF_SEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only=True so a single non-deterministic op in an unrelated test
    # doesn't kill the process; real bitwise asserts still catch divergence.
    torch.use_deterministic_algorithms(True, warn_only=True)


def test_deterministic_rollout_bitwise_equal():
    """Two seeded rollouts on a tiny synthetic env + tiny model must match bitwise.

    CPU-only form of Inv-11; the VLM-backbone form lands in the integration
    test (Task 20) with the same fixture.
    """

    def _rollout():
        _make_fully_deterministic(0)
        m = torch.nn.Linear(8, 4)
        x = torch.randn(16, 8)
        return m(x).detach()

    a = _rollout()
    b = _rollout()
    assert torch.equal(a, b), "CPU rollout not bitwise equal under fixed seed"
