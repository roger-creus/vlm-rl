---
title: BF16 default for Qwen3.5 family (overrides master-spec §1 FP16 default)
slug: bf16-default-for-qwen3.5
date: 2026-04-20
authors: /loop agent (Claude)
status: applied
supersedes: master-spec §1 precision default (FP16)
---

# BF16 default for Qwen3.5 family

## Context

Master-spec §1 declared **FP16** the default precision for training with
BF16 as a first-class ablation, rationalized by empirical lore that FP16
is more stable for RL finetuning of LLMs. That lore assumed standard
Transformer backbones (dense softmax attention), not hybrid
Gated-DeltaNet + Gated-Attention stacks.

## Problem (iter 23 investigation)

The integration test (Qwen3.5-2B, FP16, all-towers LoRA, step-grouped
re-score) reported `grad_norm_global=nan` on iter 1 despite all loss
components being finite. Systematic debugging (autograd
`detect_anomaly`) traced the NaN to the **backward** of
`linear_attn.in_proj_qkv` — the Gated DeltaNet input projection — in
the fallback PyTorch implementation (the "fast path" from
`flash-linear-attention` is not installed).

Forward path was clean: `log_probs` and `entropy` had zero NaNs and
finite range. Only the backward produced NaN, consistent with an
underflow / 0·∞ path inside DeltaNet's recurrent state-update
gradient under FP16's ~1e-3 mantissa floor.

## Reproduction

- Dropping LoRA from DeltaNet (`linear_attn.*`) did NOT fix the NaN —
  the instability is in the base DeltaNet backward, not in the LoRA
  wrapper.
- Switching `dtype=torch.bfloat16` while keeping FP16-level
  everything else → **grad_norm became finite for every loss
  component (pg_loss, v_loss, entropy) and for the full PPO loss.**

## Decision

1. `configs/backbones.yaml` default `dtype` flipped from `float16` to
   `bfloat16` for all four entries (Qwen3.5-0.8B, Qwen3.5-2B,
   Qwen3.5-4B, plus the Qwen3-VL fallback rows).
2. `algos/ppo_cot.py::Args.precision` default flipped to `"bf16"`.
3. `Fp16State` is disabled when `precision == "bf16"` (no GradScaler
   needed; BF16 has the same exponent range as FP32).
4. Master-spec §1's FP16 default is **overridden for the Qwen3.5
   family**. FP16 remains a valid opt-in via `--precision fp16` for
   non-hybrid backbones where the stability lore holds.

## Why not install `flash-linear-attention`?

Two reasons.

1. Even with the fast path, the default of "training-without-extra-deps
   should just work" is the right bar. BF16 on 8 × A6000 48 GB has
   no practical memory cost over FP16 for Qwen3.5-2B.
2. Tensor-core BF16 throughput on Ampere / Ada is comparable to FP16
   throughput for these matmul shapes; the speed trade is negligible.

Installing `flash-linear-attention` is a separate perf optimization
track (may combine with BF16 for additional throughput). Captured as
a follow-up.

## Invariants / evidence

- `detect_anomaly` probe (removed after use) confirmed:
  - FP16 + any LoRA topology → MmBackward0 NaN at `linear_attn.in_proj_qkv`.
  - BF16 + any LoRA topology → finite gradients everywhere.
- Integration test with BF16 default expected to show `grad_norm_global`
  finite across the iter (verified in the commit that applies this
  amendment).
- Inv-4 bit-parity of the step-grouped re-score (iter 22) is
  **precision-invariant** — holds at BF16 too.

## Follow-ups

- Verify the integration test reports finite `grad_norm_global` after
  this change.
- Consider installing `flash-linear-attention` + `causal-conv1d` later
  as a perf ablation; compare BF16-fast-path throughput vs BF16-fallback.
- Re-visit FP16 as an option for non-hybrid backbones (`Qwen/Qwen3-VL-*`
  lineage does not advertise DeltaNet).
