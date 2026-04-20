"""Inv-3 — ``active_adapter`` ctxmgr tripwire."""

from __future__ import annotations

import pytest

from cleanrl_vlm.models.actor_critic import active_adapter
from tests.invariants._tiny_vlm import build_tiny_ac_model

pytestmark = pytest.mark.tier1


def test_ctxmgr_asserts_expected_adapter():
    ac = build_tiny_ac_model()
    ac.vlm.model.set_adapter("actor")
    with active_adapter(ac, "actor"):
        pass


def test_ctxmgr_raises_when_adapter_mismatches():
    ac = build_tiny_ac_model()
    ac.vlm.model.set_adapter("critic")
    with pytest.raises(AssertionError), active_adapter(ac, "actor"):
        pass


def test_ctxmgr_raises_when_adapter_mutated_inside():
    ac = build_tiny_ac_model()
    ac.vlm.model.set_adapter("actor")
    with pytest.raises(AssertionError), active_adapter(ac, "actor"):
        ac.vlm.model.set_adapter("critic")
