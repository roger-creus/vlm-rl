"""Inv-5 — global grad-norm cross-check."""

import pytest
import torch

pytestmark = pytest.mark.tier1


def test_manual_vs_clip_grad_norm_agree():
    from cleanrl_vlm.training.invariants import InvariantResult, check_inv_05_grad_norm

    m = torch.nn.Linear(8, 4)
    x = torch.randn(4, 8)
    (m(x).sum()).backward()

    res = check_inv_05_grad_norm({"params": list(m.parameters())})
    assert isinstance(res, InvariantResult)
    assert res.status == "green", res.detail
