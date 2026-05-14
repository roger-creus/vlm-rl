"""Inv-6 — FP16 GradScaler stability.

This invariant checks that GradScaler is NOT repeatedly halving its scale
factor without recovery, which signals a real numerical instability. Scaled
gradients on their own may legitimately overflow to inf — GradScaler is
designed to detect that internally and skip those optimizer updates. Our test
verifies:

1. ``Fp16State.step`` runs without exception.
2. ``scale_history`` receives a new entry per ``step`` call.
3. After a handful of clean (loss = 0) steps, the scale factor has NOT
   collapsed toward zero.
4. Disabled mode bypasses scaling (current_scale == 1.0, scale is identity).
"""

import pytest
import torch

pytestmark = pytest.mark.tier1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GradScaler")
def test_fp16_state_records_scale_history_on_clean_steps():
    from cleanrl_vlm.training.precision import Fp16State

    # fp32 linear + fp32 loss → scaled grads are well-behaved (won't overflow at the
    # default GradScaler init scale). This isolates the "wrapper tracks history"
    # invariant from the separate "scaled fp16 grads may overflow" reality.
    lin = torch.nn.Linear(4, 4).cuda()
    opt = torch.optim.SGD(lin.parameters(), lr=0.01)
    fp = Fp16State(enabled=True)

    initial_scale = fp.current_scale()
    for _ in range(3):
        opt.zero_grad()
        x = torch.zeros(2, 4, device="cuda")  # zero input → zero loss → zero grad, no overflow
        y = lin(x).sum()
        fp.scale(y).backward()
        fp.step(opt)

    assert len(fp.scale_history) == 3
    assert all(s > 0 for s in fp.scale_history)
    # No collapse: after 3 clean steps the scale should NOT have halved repeatedly.
    assert fp.current_scale() >= initial_scale / 4.0


def test_fp16_state_disabled_no_scaling_applied():
    from cleanrl_vlm.training.precision import Fp16State

    fp = Fp16State(enabled=False)
    t = torch.tensor(2.0)
    assert fp.scale(t).item() == 2.0
    assert fp.current_scale() == 1.0
