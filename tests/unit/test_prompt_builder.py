import pytest


def test_prompt_builder_loads_vizdoom_basic_templates():
    from cleanrl_vlm.prompts.builder import PromptBuilder

    pb = PromptBuilder("VizdoomBasic-v1", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    actor = pb.actor_prompt()
    critic = pb.critic_prompt()
    assert "MOVE_LEFT" in actor
    assert "MOVE_RIGHT" in actor
    assert "ATTACK" in actor
    assert "ACTION: <NAME>" in actor
    assert "value estimator" in critic


def test_prompt_builder_rejects_unknown_env():
    from cleanrl_vlm.prompts.builder import PromptBuilder

    with pytest.raises(KeyError):
        PromptBuilder("Unknown-v0", ["A", "B"])
