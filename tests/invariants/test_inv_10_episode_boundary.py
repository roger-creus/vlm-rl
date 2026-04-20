"""Inv-10 — GAE resets exactly at ``done=True`` boundaries."""

import pytest
import torch

pytestmark = pytest.mark.tier1


def test_gae_resets_at_done_boundary():
    from cleanrl_vlm.rollout.buffer import compute_gae

    rewards = torch.tensor([[1.0], [2.0], [10.0], [20.0]])
    values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # episode 1 ends at step 1
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])
    gamma, lam = 0.99, 0.95

    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)

    # At step 1 (done=True), A[1] should equal r[1] exactly (boot from next = 0)
    # and advantage at step 0 must NOT include step 2's reward.
    # Advantage at step 0 with done[0]=0: A[0] = r[0] + gamma*lam*(1-done[0])*A[1]
    expected_a1 = 2.0
    expected_a0 = 1.0 + gamma * lam * (1 - 0) * expected_a1
    assert torch.allclose(adv[0], torch.tensor([expected_a0]), atol=1e-5)
    assert torch.allclose(adv[1], torch.tensor([expected_a1]), atol=1e-5)

    # Episode 2: steps 2 and 3. A[2] must NOT be polluted by episode 1 rewards.
    a3 = 20.0
    a2 = 10.0 + gamma * lam * (1 - 0) * a3
    assert torch.allclose(adv[2], torch.tensor([a2]), atol=1e-5)
