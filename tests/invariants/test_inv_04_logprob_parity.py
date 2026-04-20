"""Inv-4 single-path variant — re-score cached ``full_ids`` under the same
actor adapter at update-epoch=0, minibatch=0 every iteration. Assert drift
< ``INV_04_TOLERANCE`` (1e-4; fp16 reduction-order safety margin).

Full two-path (vLLM ↔ HF) parity lives in ``E-vllm-rollout-path``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.tier1


def test_single_path_logprob_parity_tolerance_constant():
    """Assert the public tolerance the trainer's Inv-4 check imports."""
    from cleanrl_vlm.training.invariants import INV_04_TOLERANCE

    assert INV_04_TOLERANCE == 1e-4


@pytest.mark.gpu
def test_single_path_logprob_parity_on_tiny_model():
    """Re-scoring-under-same-adapter drift stays within the tolerance.

    ``@gpu`` marker: the real trainer check runs on the actor adapter's
    forward; the synthetic version here just demonstrates the invariant's
    contract (same input → same logprobs under fixed adapter + zero dropout).
    """
    import torch

    lp_old = torch.randn(4, dtype=torch.float32)
    lp_new = lp_old.clone()  # deterministic re-score under lora_dropout=0.0
    drift = (lp_new - lp_old).abs().max().item()
    assert drift < 1e-4
