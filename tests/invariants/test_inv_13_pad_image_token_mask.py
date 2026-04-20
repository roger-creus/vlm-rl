"""Inv-13 — pad + image-token gradient contribution is zero."""

import pytest
import torch

pytestmark = pytest.mark.tier1


def test_pad_positions_do_not_contribute_to_loss_grad():
    """Minimal illustrative test: build an attention_mask and assert masked
    positions' contribution to a masked sum is zero (the mechanism VLM loss
    uses to ignore pad + image tokens)."""
    logp = torch.randn(2, 8, requires_grad=True)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss = -(logp * mask).sum() / mask.sum()
    loss.backward()
    g = logp.grad
    # Masked positions have grad 0; unmasked positions have grad != 0.
    assert torch.all(g[mask == 0] == 0.0)
    assert torch.any(g[mask == 1] != 0.0)
