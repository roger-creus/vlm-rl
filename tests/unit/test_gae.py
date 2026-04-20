import torch


def test_gae_matches_hand_computed():
    from cleanrl_vlm.rollout.buffer import compute_gae

    # 3 steps, 1 env
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.3], [0.1]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])
    gamma, lam = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)

    # Hand compute
    d2 = 2.0 + gamma * 0.0 * (1 - 0) - 0.1
    d1 = 0.0 + gamma * 0.1 * (1 - 0) - 0.3
    d0 = 1.0 + gamma * 0.3 * (1 - 0) - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * (1 - 0) * a2
    a0 = d0 + gamma * lam * (1 - 0) * a1

    assert torch.allclose(adv[0], torch.tensor([a0]), atol=1e-5)
    assert torch.allclose(adv[1], torch.tensor([a1]), atol=1e-5)
    assert torch.allclose(adv[2], torch.tensor([a2]), atol=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-5)
