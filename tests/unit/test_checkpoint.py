def test_save_signature_accepts_algo_slug_kwarg():
    """Reviewer m6: function signature must take algo_slug parameter."""
    import inspect

    from cleanrl_vlm.training.checkpoint import save_vlm_actor_critic_checkpoint

    sig = inspect.signature(save_vlm_actor_critic_checkpoint)
    assert "algo_slug" in sig.parameters
    assert "ac_model" in sig.parameters
    assert "optimizer" in sig.parameters
