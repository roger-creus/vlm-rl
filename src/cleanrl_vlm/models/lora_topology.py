"""LoRA target-module group resolution."""

from __future__ import annotations

from collections.abc import Iterable

_GROUPS: dict[str, list[str]] = {
    "text_attn": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
    "text_mlp": [
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    "vision_attn": [
        "attn.qkv",
        "attn.proj",
    ],
    "vision_mlp": [
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    ],
    "merger": [
        "merger.linear_fc1",
        "merger.linear_fc2",
    ],
    "lm_head": [
        "lm_head",
    ],
    "text_moe": [
        "block_sparse_moe.gate",
    ],
}


def default_target_modules(groups: Iterable[str]) -> list[str]:
    """Resolve a set of LoRA group names to a flat de-duplicated list of
    module-name suffixes consumable by ``peft.LoraConfig(target_modules=...)``.

    Raises ValueError on unknown group names.
    """
    groups = list(groups)
    unknown = [g for g in groups if g not in _GROUPS]
    if unknown:
        raise ValueError(f"Unknown LoRA group(s): {unknown}. Known: {sorted(_GROUPS)}")
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for m in _GROUPS[g]:
            if m not in seen:
                out.append(m)
                seen.add(m)
    return out
