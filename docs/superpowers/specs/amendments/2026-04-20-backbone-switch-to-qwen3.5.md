---
title: Backbone switch to Qwen3.5 family
slug: backbone-switch-to-qwen3.5
date: 2026-04-20
authors: /loop agent (Claude)
status: applied
supersedes: 2026-04-20-backbone-names-correction.md (partial)
---

# Backbone switch to Qwen3.5 family

## Context

User directive (mid-iter 20) to switch the primary backbone from
`Qwen/Qwen3-VL-2B-Instruct` to `Qwen/Qwen3.5-2B`, with
`Qwen/Qwen3.5-0.8B` as the smaller debug model. The 2026-04-20 backbone
names correction amendment observed that "Qwen3.5-VL" did not exist on
HF at that time; as of the user's check the `Qwen/Qwen3.5-{0.8,2,4}B`
models have been published and **do** carry a vision encoder despite
the absence of `-VL-` in the slug. They register under
`AutoModelForImageTextToText` and expose `vision_config` + `text_config`.

This amendment brings the library back in line with the master spec
§3's original intent (hybrid Gated-DeltaNet + Gated-Attention + sparse
MoE backbones).

## Architecture confirmation

Module introspection on `Qwen/Qwen3.5-2B`:

- **Text tower**: alternating Gated-Attention blocks
  (`self_attn.q/k/v/o_proj`) and Gated DeltaNet blocks
  (`linear_attn.in_proj_{a,b,qkv,z}`, `linear_attn.out_proj`),
  SwiGLU MLPs (`mlp.gate_proj` / `up_proj` / `down_proj`), `lm_head`
  (tied by default).
- **Vision tower**: ViT blocks (`attn.qkv` fused + `attn.proj`),
  MLPs `mlp.linear_fc1` / `linear_fc2`, patch embed.
- **Merger**: `merger.linear_fc1` / `linear_fc2`.

## Changes applied this iter

1. `configs/backbones.yaml` — primary entries are now Qwen3.5-0.8B,
   Qwen3.5-2B (Tier-1), Qwen3.5-4B (Tier-2).
2. `src/cleanrl_vlm/models/lora_topology.py` — `text_attn` group
   extended with the DeltaNet linear-attention projections so LoRA
   covers both attention-block types. New `ALL_TOWERS_DEFAULT` tuple
   for the explicit "all vision and language towers" default (per
   user directive).
3. `configs/backbones.yaml::lora_groups_default` = all 6 groups
   (text_attn, text_mlp, vision_attn, vision_mlp, merger, lm_head).
4. `algos/ppo_cot.py::Args.backbone` default → `Qwen/Qwen3.5-2B`.
   `Args.lora_groups` default → all 6 groups.
5. `tests/smoke/test_hello_vlm.py` default `MODEL_ID` → `Qwen/Qwen3.5-2B`.
6. `scripts/probe_backbone.py` re-run on Qwen3.5-2B @ 76800 px;
   `docs/backbone_probes/qwen3.5-2b.md` written (280 image tokens,
   Inv-8 PASS).
7. New `tests/integration/test_qwen35_backbone_wiring.py` covers:
   - image tokens present (Inv-8 smoke),
   - every LoRA group wraps ≥1 module on the real backbone,
   - Inv-1 trainability split under PEFT,
   - Inv-3 base-weight identity across adapters + ctxmgr tripwire,
   - actor/critic param-id disjointness.
8. `src/cleanrl_vlm/models/actor_critic.py::_adapter_param_ids` — no
   longer filters by `requires_grad`, because PEFT's `set_adapter`
   flips that flag on the non-active adapter and made the disjointness
   assertion dependent on current state (caught by the new wiring
   test). Semantics are now about parameter **identity**, not mutable
   trainable flags.

## Verification

- `ruff check .` + `ruff format --check .` clean.
- `mkdocs build --strict` clean.
- `pytest tests/unit tests/invariants -m "not gpu"` — 49/49 pass.
- `pytest tests/smoke/test_hello_vlm.py -m "tier1 and gpu"` — PASS in
  45.94 s (Qwen3.5-2B loads, generates, decodes).
- `pytest tests/integration/test_qwen35_backbone_wiring.py
  -m "tier1 and gpu"` — 7/7 pass in 65.67 s.
- `pytest tests/integration/test_trainer_short_run.py
  -m "tier1 and gpu"` — PASS in 76.83 s. Full PPO-COT pipeline runs
  end-to-end against the new backbone + all-towers LoRA.

## Known signal (per §0 — investigate but not hard-fail)

On the integration test, `inv_4_status=red` on iter 1 — first-minibatch
first-epoch drift is ~8 × 10⁻³, larger than the 1e-4 tolerance.
Isolated debug (`scripts/_debug_parity.py`, now removed) confirms
**exact** parity (drift = 0.0) when re-scoring a single rolled-out
row. The integration test batches two rows from different rollout
steps and re-scores them in a single batch=2 forward; the drift is
likely fp16 kernel/reduction noise across 24 layers at batch=2 that
isn't present in the batch=1 rollout forwards.

Investigation options for a future iter:
- Run re-score row-by-row with batch=1 to match rollout semantics
  (simplest; kills the noise source at the cost of wall-clock).
- Retain batched re-score and raise `INV_04_TOLERANCE` with evidence
  that the drift stays bounded (say 1e-2) across many iters without
  surviving to a real correctness bug.
- Switch to BF16 — larger exponent, less reduction-order noise —
  per master-spec §1 this is the first-class ablation.

## Resolution (iter 21)

Root-caused and fixed. A targeted diagnostic probe (since removed)
ran 2 rollout steps at `num_envs=1` then re-scored each cached row
both **alone** at batch=1 and **together** at batch=2 (padded to
S_max). Observed:

- Solo re-score: drift = `0.0` for **both** rows (bit-exact).
- Batched re-score: the **shorter** row (padded with `pad_id` to
  S_max) has drift = `2.93e-02`. The longer row (no trailing pad)
  has drift = `0.0`.

So the noise is not "batch=2 in general" — it is specifically the
*padded* row whose forward passes through a padding-aware kernel
path whose fp16 numerics differ from the rollout's batch=1 no-pad
forward. The padded-row bias is not a reporting nuisance — it would
bias the PPO ratio and therefore the policy gradient, a real
correctness defect.

**Fix** (iter 21, commit following this amendment): PPO update
re-scores row-by-row at batch=1, each call with the row's own
`full_ids` and `prompt_lens`. This matches rollout semantics exactly.
The integration test's new `inv_4_status == "green"` assertion
reproduces the bug pre-fix (drift ~8e-3) and passes post-fix
(drift = 0.0). Wall-clock cost is `minibatch_size × forward_time`
instead of `1 × batched_forward`; acceptable for Tier-1. Tier-2
optimization can revisit with a length-bucketing scheme later.
