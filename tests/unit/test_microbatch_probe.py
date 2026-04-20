def test_probe_microbatch_returns_largest_non_ooming_size():
    from cleanrl_vlm.training.microbatch_probe import probe_microbatch

    calls = {"count": 0}

    def try_batch(size: int) -> bool:
        calls["count"] += 1
        # Pretend sizes 1, 2, 4 succeed; 8 OOMs.
        return size <= 4

    picked = probe_microbatch(try_batch_fn=try_batch, cap=32)
    assert picked == 4
    assert calls["count"] >= 4  # tried 1, 2, 4, 8 (OOM)


def test_probe_microbatch_respects_cap():
    from cleanrl_vlm.training.microbatch_probe import probe_microbatch

    picked = probe_microbatch(try_batch_fn=lambda s: True, cap=16)
    assert picked == 16


def test_probe_microbatch_returns_1_when_even_1_fails():
    from cleanrl_vlm.training.microbatch_probe import probe_microbatch

    picked = probe_microbatch(try_batch_fn=lambda s: False, cap=16)
    assert picked == 1
