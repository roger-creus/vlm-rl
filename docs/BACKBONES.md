# Backbones

## Probed

| backbone | params | context | probe |
|---|---|---|---|
| `Qwen/Qwen3.5-2B` | 2 B | 262K | [report](backbone_probes/qwen3.5-2b.md) (Inv-8 PASS: 280 image tokens @ 320×240 / 76800 px) |
| `Qwen/Qwen3-VL-2B-Instruct` | 2 B | 262K | [report](backbone_probes/qwen3-vl-2b-instruct.md) (Inv-8 PASS: 280 image tokens @ 320×240 / 76800 px) |

## Current roster

Both the Qwen3.5 family (master-spec §3's original target) and the
Qwen3-VL family exist on Hugging Face. We prefer Qwen3.5 for new work —
it is the hybrid Gated-DeltaNet + Gated-Attention + MoE architecture
the master spec anticipated, and it ships as an instruct model with
a native chat template.

| name | params | thinking default | tier | notes |
|------|--------|------------------|------|-------|
| `Qwen/Qwen3.5-0.8B` | 0.8 B | off | Tier-1 debug / fast-smoke | smallest hybrid, matches spec §3's 0.8B slot |
| `Qwen/Qwen3.5-2B` | 2 B | off | **Tier-1 primary** | default in `algos/ppo_cot.py` |
| `Qwen/Qwen3.5-4B` | 4 B | off | Tier-2 paper runs | spec's target 4B size |
| `Qwen/Qwen3-VL-2B-Instruct` | 2 B | off | fallback | retained for comparison runs |
| `Qwen/Qwen3-VL-4B-Instruct` | 4 B | off | fallback | retained for comparison runs |

All load via `AutoModelForImageTextToText` + `AutoProcessor`. Apache-2.0.
Require `transformers` from git main (≥ 5.6.0.dev0 registers `qwen3_5`).

## Architecture (Qwen3.5)

Confirmed by module introspection on `Qwen/Qwen3.5-2B`:

- **Text tower**: alternating Gated-Attention (`self_attn.q/k/v/o_proj`) and
  Gated DeltaNet linear-attention (`linear_attn.in_proj_{a,b,qkv,z}`,
  `linear_attn.out_proj`) blocks. SwiGLU MLPs (`mlp.gate_proj`,
  `mlp.up_proj`, `mlp.down_proj`).
- **Vision tower**: ViT-like with `attn.qkv` (fused) + `attn.proj`,
  MLPs `mlp.linear_fc1` / `linear_fc2`, patch embed + 2D position embed.
- **Merger**: vision→text projection via `merger.linear_fc1` /
  `merger.linear_fc2`.
- **Shared**: `lm_head` (tied to input embedding by default).

The `lora_groups_default` in `configs/backbones.yaml` wraps **all six
groups** (text_attn / text_mlp / vision_attn / vision_mlp / merger /
lm_head) to put LoRA on every trainable surface.

## Onboarding ritual

Per master-spec §11 S-6: registry entry + processor probe + §4 image-input
probes + Inv-1..Inv-14 pass + `docs/backbone_probes/<name>.md`.

The mechanical tests for the active backbone live at
`tests/integration/test_qwen35_backbone_wiring.py` — image-token
presence, LoRA module-hit per group, Inv-1/3 on the real model, and
actor/critic adapter disjointness.
