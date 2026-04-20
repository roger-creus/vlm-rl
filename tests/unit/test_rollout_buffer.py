import torch


def test_buffer_allocates_correct_shapes():
    from cleanrl_vlm.rollout.buffer import RolloutBuffer

    rb = RolloutBuffer(num_envs=4, num_steps=32, obs_shape=(3, 240, 320), device=torch.device("cpu"))
    assert rb.obs.shape == (32, 4, 3, 240, 320)
    assert rb.actions.shape == (32, 4)
    assert rb.logprob_sum.shape == (32, 4)
    assert rb.rewards.shape == (32, 4)
    assert rb.values.shape == (32, 4)
    assert rb.dones.shape == (32, 4)


def test_buffer_dtypes():
    from cleanrl_vlm.rollout.buffer import RolloutBuffer

    rb = RolloutBuffer(num_envs=2, num_steps=4, obs_shape=(3, 8, 8), device=torch.device("cpu"))
    assert rb.obs.dtype == torch.uint8
    assert rb.actions.dtype == torch.long
    assert rb.logprob_sum.dtype == torch.float32
    assert rb.rewards.dtype == torch.float32
    assert rb.values.dtype == torch.float32
    assert rb.dones.dtype == torch.float32
