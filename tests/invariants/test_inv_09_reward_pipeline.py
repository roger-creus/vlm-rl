"""Inv-9 — reward pipeline integrity."""

import pytest
import torch

pytestmark = pytest.mark.tier1


def test_scripted_rewards_flow_into_gae_unchanged():
    from cleanrl_vlm.rollout.buffer import compute_gae

    rewards = torch.tensor([[0.0], [1.0], [0.0], [2.0]])
    values = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards)
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])

    adv, _ret = compute_gae(rewards, values, dones, next_value, next_done, gamma=1.0, lam=1.0)
    # With gamma=1, lam=1, V=0: advantage[t] = sum_{u>=t} r[u].
    assert torch.allclose(adv[0], torch.tensor([3.0]))
    assert torch.allclose(adv[1], torch.tensor([3.0]))
    assert torch.allclose(adv[2], torch.tensor([2.0]))
    assert torch.allclose(adv[3], torch.tensor([2.0]))
