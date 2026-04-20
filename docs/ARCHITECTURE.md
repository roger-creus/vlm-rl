# Architecture

Placeholder — populated by /loop per master-spec §12 rituals as the library fills in.

## Planned contents

- Library layout (mirrors master-spec §2).
- Rollout / train split: vLLM serves actor LoRA for COT generation; HF+PEFT+DeepSpeed (or FSDP2) handles the gradient step.
- LoRA dual-adapter pattern (`actor` + `critic` on the same base VLM; swap via `set_adapter`).
- Configurable LoRA target-module groups (text_attn, text_mlp, text_moe, vision_attn, vision_mlp, merger, lm_head).
- Hybrid Gated-DeltaNet + Gated-Attention + sparse MoE specifics for the Qwen3.5-VL backbones.
- Where each §8 invariant lives (unit test + `InvariantMonitor` hook).
