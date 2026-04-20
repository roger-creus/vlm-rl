import pytest


def test_default_target_modules_text_attn():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    out = default_target_modules({"text_attn"})
    assert "self_attn.q_proj" in out
    assert "self_attn.k_proj" in out
    assert "self_attn.v_proj" in out
    assert "self_attn.o_proj" in out
    assert "mlp.gate_proj" not in out


def test_default_target_modules_text_mlp():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    out = default_target_modules({"text_mlp"})
    assert set(out) >= {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def test_default_target_modules_lm_head():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    out = default_target_modules({"lm_head"})
    assert "lm_head" in out


def test_default_target_modules_combined():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    out = default_target_modules({"text_attn", "text_mlp", "lm_head"})
    assert "self_attn.q_proj" in out
    assert "mlp.down_proj" in out
    assert "lm_head" in out


def test_default_target_modules_unknown_group_raises():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    with pytest.raises(ValueError):
        default_target_modules({"bogus"})


def test_default_target_modules_returns_list_without_duplicates():
    from cleanrl_vlm.models.lora_topology import default_target_modules

    out = default_target_modules({"text_attn"})
    assert isinstance(out, list)
    assert len(out) == len(set(out))
