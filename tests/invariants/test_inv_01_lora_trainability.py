"""Inv-1 — LoRA trainability split.

Asserts:
1. requires_grad == True iff "lora_" in name or param ∈ critic_head.
2. Base weights keep identity (``id(p)``) across ``set_adapter`` swaps.
3. Optimizer param groups for actor and critic are disjoint.
"""

from __future__ import annotations

import pytest

from tests.invariants._tiny_vlm import build_tiny_ac_model

pytestmark = pytest.mark.tier1


def test_inv_01_requires_grad_split():
    ac = build_tiny_ac_model()
    for n, p in ac.vlm.model.named_parameters():
        expected = "lora_" in n
        assert p.requires_grad == expected, f"param {n} requires_grad={p.requires_grad} expected {expected}"
    for p in ac.critic_head.parameters():
        assert p.requires_grad is True


def test_inv_01_base_weight_identity_across_set_adapter():
    ac = build_tiny_ac_model()
    ac.vlm.model.set_adapter("actor")
    base_ids_actor = {n: id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" not in n}
    ac.vlm.model.set_adapter("critic")
    base_ids_critic = {n: id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" not in n}
    assert base_ids_actor == base_ids_critic, (
        "base-weight tensor ids changed across set_adapter('actor') vs ('critic') -- "
        "adapters are NOT sharing base weights"
    )


def test_inv_01_actor_critic_param_groups_disjoint():
    ac = build_tiny_ac_model()
    actor_ids = ac.actor_param_ids()
    critic_ids = ac.critic_param_ids()
    assert actor_ids, "no trainable actor params found"
    assert critic_ids, "no trainable critic params found"
    assert actor_ids.isdisjoint(critic_ids), (
        f"actor and critic param groups overlap on {len(actor_ids & critic_ids)} tensors"
    )
