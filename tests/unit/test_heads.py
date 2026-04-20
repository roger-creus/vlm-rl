import torch


def test_critic_head_shape():
    from cleanrl_vlm.models.heads import CriticHead

    h = CriticHead(input_dim=128)
    x = torch.randn(4, 128)
    y = h(x)
    assert y.shape == (4, 1)


def test_actor_head_shape():
    from cleanrl_vlm.models.heads import ActorHead

    h = ActorHead(input_dim=128, num_actions=7)
    x = torch.randn(4, 128)
    y = h(x)
    assert y.shape == (4, 7)


def test_layer_init_is_orthogonal():
    import torch.nn as nn

    from cleanrl_vlm.models.heads import layer_init

    lin = layer_init(nn.Linear(32, 16))
    w = lin.weight.detach()
    gram = w @ w.t()
    # Orthogonal rows → gram matrix is diagonal (off-diagonals zero to fp tolerance).
    assert torch.allclose(gram, gram.diag().diag(), atol=1e-4)
    assert torch.allclose(lin.bias.detach(), torch.zeros_like(lin.bias))
