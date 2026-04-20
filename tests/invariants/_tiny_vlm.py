"""Tiny VLM fixture used by invariant tests.

Registers a minimal nn.Module tree that PEFT can LoRA-wrap so the Inv-01/03
tests can run on CPU without loading the real 2B backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class _TextConfig:
    hidden_size = 32
    vocab_size = 128


class _TinyConfig:
    def __init__(self) -> None:
        self.text_config = _TextConfig()


class _SelfAttn(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _Block(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.self_attn = _SelfAttn(d)


class _TinyModel(nn.Module):
    """Pure-CPU stub with the minimal attribute surface ActorCritic touches."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.block = _Block(32)
        self.lm_head = nn.Linear(32, 128, bias=False)


def build_tiny_ac_model():
    """Return a stub that mimics ``DecoupledActorCriticVLM_COT``'s attribute
    surface enough to exercise Inv-01 / Inv-03 tests.

    Bypasses ``BaseVLM.from_pretrained`` (no HF / no CUDA). Exposes
    ``.vlm.model`` (the PEFT-wrapped TinyModel), ``.critic_head``,
    ``.actor_param_ids()``, ``.critic_param_ids()``.
    """
    from cleanrl_vlm.models.heads import CriticHead

    inner = _TinyModel()
    base_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "lm_head",
        ],
        bias="none",
    )
    peft_model = get_peft_model(inner, base_cfg, adapter_name="actor")
    peft_model.add_adapter("critic", base_cfg)

    class _StubVLM:
        def __init__(self, m: nn.Module) -> None:
            self.model = m

    class _StubAC:
        pass

    ac = _StubAC()
    ac.vlm = _StubVLM(peft_model)
    ac.critic_head = CriticHead(32).to(torch.float32)
    for n, p in ac.vlm.model.named_parameters():
        p.requires_grad = "lora_" in n
    for p in ac.critic_head.parameters():
        p.requires_grad = True

    def actor_ids() -> set[int]:
        return {id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" in n and ".actor." in n}

    def critic_ids() -> set[int]:
        s = {id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" in n and ".critic." in n}
        s |= {id(p) for p in ac.critic_head.parameters()}
        return s

    ac.actor_param_ids = actor_ids
    ac.critic_param_ids = critic_ids
    return ac
