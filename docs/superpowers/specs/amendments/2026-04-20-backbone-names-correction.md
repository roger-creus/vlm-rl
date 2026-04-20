---
title: Backbone names correction — Qwen3-VL (not Qwen3.5-VL)
date: 2026-04-20
authors: Claude (/loop agent)
amends: 2026-04-19-cleanrl-vlm-masterplan.md (§3, §14, Appendix C)
rationale: reality check against Hugging Face model registry
---

# Amendment: backbone names correction

## Finding

During iter 3 `A2-bootstrap-finalize`, the `/loop` agent attempted to
resolve the two backbones named in master-spec §3:

- `Qwen/Qwen3.5-VL-0.8B` — Tier-1 smoke + debug.
- `Qwen/Qwen3.5-VL-4B` — Tier-2 paper runs.

Both **do not exist** on Hugging Face as of 2026-04-20 (`HfApi().model_info`
raises `RepositoryNotFoundError`). An HF search for `author=Qwen, search=VL`
returns the actual current families:

- `Qwen/Qwen2-VL-{2B,7B,72B}-Instruct`
- `Qwen/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct`
- `Qwen/Qwen3-VL-{2B,4B,8B,30B-A3B,235B-A22B}-Instruct` (plus -Thinking
  / -FP8 / -GGUF / embedding / reranker variants)

There is no `Qwen3.5-VL` series published. The master spec's Appendix C
also points at `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-0.8B` URLs — neither
of these repos exists either.

## Adaptation (per §11 S-11 "spec evolves" + §13.1 "pick spec-aligned default")

Spec intent: use the latest Qwen VL family for the hero experiments and
a smaller sibling for Tier-1 smoke. Mapping to reality:

| Spec tier | Spec ID (not available) | **Use instead**                 | Params | Thinking default |
|-----------|-------------------------|----------------------------------|--------|------------------|
| Tier-1 smoke   | `Qwen/Qwen3.5-VL-0.8B` | `Qwen/Qwen3-VL-2B-Instruct`      | 2 B    | off (`-Instruct`) |
| Tier-2 hero    | `Qwen/Qwen3.5-VL-4B`   | `Qwen/Qwen3-VL-4B-Instruct`      | 4 B    | off (`-Instruct`) |
| Tier-2 thinking ablation | — (not in spec)      | `Qwen/Qwen3-VL-4B-Thinking`      | 4 B    | **on**            |
| Tier-2 scale-up ablation | — (not in spec)      | `Qwen/Qwen3-VL-8B-Instruct`      | 8 B    | off               |

No 0.8B variant exists — 2B-Instruct replaces it as the Tier-1 default.
Tier-2 hero stays 4 B as the spec prescribes.

## Architecture clarification

Master-spec §3 describes Qwen3.5-VL as "hybrid Gated-DeltaNet +
Gated-Attention + sparse MoE". Qwen3-VL does **not** advertise the same
hybrid architecture on its model cards (spot-checked 2026-04-20) —
appears to be a standard transformer with vision encoder + merger +
decoder. The /loop agent will:

1. Web-fetch each Qwen3-VL model card per S-5 the first time a trainer
   loads the backbone.
2. Record the actual architecture layout in `docs/backbone_probes/<name>.md`.
3. Update `docs/BACKBONES.md` and this amendment with the observed
   details.

Any spec text that depends on the hybrid architecture (Gated-DeltaNet /
MoE LoRA groups in §3's LoRA-topology table) should be re-read as
"conditional on the architecture carrying those blocks" — if Qwen3-VL
doesn't have MoE or DeltaNet, those LoRA groups are no-ops and their
ablations shrink accordingly.

## Thinking-mode defaults

Spec §3 says: "Qwen3.5-VL-4B native thinking mode on by default". Qwen3-VL
has explicit `-Instruct` vs `-Thinking` splits in the model ID (two
separate weights). This simplifies the dispatch:

- Canon trainers load `-Instruct` by default.
- The "thinking-mode COT" ablation (spec §14 paper-ablations list)
  loads `-Thinking`.

No dynamic thinking toggle needed inside a single backbone; the
choice is materialized as which weights to load.

## Downstream file updates this amendment triggers

Immediate (iter 3):

- `tests/smoke/test_hello_vlm.py`: default `MODEL_ID` env var
  `CLEANRL_VLM_SMOKE_MODEL` → `Qwen/Qwen3-VL-2B-Instruct`.
- `README.md`: quickstart references `Qwen3-VL` everywhere.
- `docs/BACKBONES.md`: replace the two-row table with the four-row
  variant above.
- `pyproject.toml` has no backbone ID hard-coded; unchanged.

Deferred (iter 4+, when each module lands):

- `configs/backbones.yaml` (per §2) — canonical per-backbone defaults
  keyed on the corrected IDs.
- `LOOP_STATE.md`'s `B-ppo-cot-vizdoom-basic-0.8B` task ID renames to
  `B-ppo-cot-vizdoom-basic-2B` (and the initial trainer targets
  `Qwen/Qwen3-VL-2B-Instruct`).
- `docs/backbone_probes/qwen3-vl-2b-instruct.md` and
  `qwen3-vl-4b-instruct.md` land the first time each backbone is
  loaded.
- Master-spec §3 text will be lightly amended in-place (next full
  spec-edit cycle) to unify the naming; this amendment stands in the
  meantime.

## Non-changes

- Tier-1 / Tier-2 split (spec §4).
- §8 invariants. Architecture-contingent ones (Inv-14 distributed
  broadcast over MoE layers) simply don't fire on non-MoE backbones
  — the test short-circuits gracefully.
- Apache-2.0 licensing assumption (Qwen3-VL also Apache-2.0).
- Native `<think>` tag handling in the COT parser — Qwen3-VL-Thinking
  uses the same tag format.

## Sign-off

Self-reviewed per §13.1 (no human approval available). Committed alongside
`tests/smoke/test_hello_vlm.py` model-ID update and `docs/BACKBONES.md`
rewrite. AUTONOMY_LOG iter-3 entry lists this amendment.
