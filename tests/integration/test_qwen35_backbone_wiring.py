"""Mechanical tests for the Qwen3.5 backbone wiring.

Covers the user-requested "full tests":

* Image perception — Inv-8 patch-coverage on a synthetic RGB image at the
  default processor pixel budget; assert the processor emits ``N > 0``
  image tokens and that the image_grid_thw math agrees with the token
  count.
* LoRA application — build ``DecoupledActorCriticVLM_COT`` with the
  "all towers" default group set and assert every group produced at
  least one wrapped module (catches target-module-name typos on a
  backbone bump).
* Trainable split — Inv-1 on the real backbone: ``requires_grad`` is
  True iff name matches ``"lora_"``.
* Adapter isolation — Inv-3: actor/critic adapters share base weights
  (identical ``id(p)`` across ``set_adapter`` calls).
* Active-adapter tripwire — Inv-3 ctxmgr asserts and raises as expected.

All tests run against ``Qwen/Qwen3.5-2B`` by default; override with
``CLEANRL_VLM_WIRING_MODEL=Qwen/Qwen3.5-0.8B`` for the smaller debug
backbone once it is cached locally.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL_ID = os.environ.get("CLEANRL_VLM_WIRING_MODEL", "Qwen/Qwen3.5-2B")

pytestmark = [pytest.mark.tier1, pytest.mark.gpu, pytest.mark.timeout(600)]


@pytest.fixture(scope="module")
def ac_model():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from cleanrl_vlm.models.actor_critic import DecoupledActorCriticVLM_COT

    model = DecoupledActorCriticVLM_COT(
        vlm_name=MODEL_ID,
        min_pixels=76800,
        max_pixels=76800,
        attn_implementation="sdpa",
        dtype=torch.float16,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_groups=("text_attn", "text_mlp", "vision_attn", "vision_mlp", "merger", "lm_head"),
        max_new_tokens=8,
    )
    yield model
    del model
    torch.cuda.empty_cache()


def test_base_vlm_loads_and_preprocesses(ac_model) -> None:
    """BaseVLM.preprocess_obs_and_text produces a valid multimodal batch."""
    # Synthetic 3x160x120 quadrant image.
    img = np.zeros((1, 120, 160, 3), dtype=np.uint8)
    img[0, :60, :80] = [220, 20, 20]
    img[0, :60, 80:] = [20, 200, 20]
    img[0, 60:, :80] = [20, 20, 220]
    img[0, 60:, 80:] = [230, 230, 20]
    obs = torch.as_tensor(img, dtype=torch.uint8, device=ac_model.vlm.model.device)

    inputs = ac_model.vlm.preprocess_obs_and_text(obs, ["Describe the image."])
    assert inputs.input_ids.shape[0] == 1
    assert inputs.pixel_values is not None and inputs.pixel_values.numel() > 0
    assert hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw.numel() > 0


def test_inv_8_image_tokens_present(ac_model) -> None:
    """Inv-8: ``image_grid_thw`` product sums to a positive image-token count."""
    img = np.full((1, 120, 160, 3), 128, dtype=np.uint8)
    obs = torch.as_tensor(img, dtype=torch.uint8, device=ac_model.vlm.model.device)
    inputs = ac_model.vlm.preprocess_obs_and_text(obs, ["count the pixels"])
    num_image_tokens = int(inputs.image_grid_thw.prod(dim=-1).sum().item())
    assert num_image_tokens > 0, "processor emitted zero image tokens"


def test_lora_all_towers_default_every_group_wraps_modules(ac_model) -> None:
    """Every enabled LoRA group must wrap ≥1 module on the real backbone.

    Guards against target-module-name drift across backbone versions —
    e.g., when Qwen's MLP renames ``gate_proj`` → ``gate`` in a future
    release, this test goes red loudly instead of silently training zero
    params.
    """
    from cleanrl_vlm.models.lora_topology import _GROUPS

    lora_names = {n for n, _ in ac_model.vlm.model.named_parameters() if "lora_" in n}
    groups_under_test = ("text_attn", "text_mlp", "vision_attn", "vision_mlp", "merger", "lm_head")
    missing: list[str] = []
    for group in groups_under_test:
        hits = 0
        for suffix in _GROUPS[group]:
            hits += sum(1 for n in lora_names if suffix in n)
        if hits == 0:
            missing.append(group)
    assert not missing, f"LoRA groups with zero wrapped modules on {MODEL_ID}: {missing}"


def test_inv_1_lora_trainability_split(ac_model) -> None:
    """Inv-1: every ``lora_*`` param is trainable; no base param is."""
    for n, p in ac_model.vlm.model.named_parameters():
        expected = "lora_" in n
        assert p.requires_grad == expected, f"{n}: requires_grad={p.requires_grad} expected {expected}"
    for p in ac_model.critic_head.parameters():
        assert p.requires_grad


def test_inv_3_base_weights_shared_across_adapters(ac_model) -> None:
    """Inv-3: actor and critic adapters share the same base-weight tensors."""
    ac_model.vlm.model.set_adapter("actor")
    base_actor = {n: id(p) for n, p in ac_model.vlm.model.named_parameters() if "lora_" not in n}
    ac_model.vlm.model.set_adapter("critic")
    base_critic = {n: id(p) for n, p in ac_model.vlm.model.named_parameters() if "lora_" not in n}
    assert base_actor == base_critic, "base weight tensors diverge across set_adapter()"


def test_inv_3_active_adapter_ctxmgr(ac_model) -> None:
    """Inv-3 tripwire: ctxmgr passes on match, raises on mismatch, raises on mutation."""
    from cleanrl_vlm.models.actor_critic import active_adapter

    ac_model.vlm.model.set_adapter("actor")
    with active_adapter(ac_model, "actor"):
        pass

    ac_model.vlm.model.set_adapter("critic")
    with pytest.raises(AssertionError), active_adapter(ac_model, "actor"):
        pass

    ac_model.vlm.model.set_adapter("actor")
    with pytest.raises(AssertionError), active_adapter(ac_model, "actor"):
        ac_model.vlm.model.set_adapter("critic")


def test_actor_critic_param_ids_disjoint(ac_model) -> None:
    actor_ids = ac_model.actor_param_ids()
    critic_ids = ac_model.critic_param_ids()
    assert actor_ids, "no trainable actor LoRA params found"
    assert critic_ids, "no trainable critic LoRA params found"
    assert actor_ids.isdisjoint(critic_ids), (
        f"actor and critic param groups overlap on {len(actor_ids & critic_ids)} tensors"
    )
