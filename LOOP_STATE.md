# LOOP_STATE.md

**Last updated:** 2026-04-20 (iter 3 — bootstrap fully finalized, hello-VLM smoke green).
**Maintained by:** the autonomous `/loop` agent; humans read only.

## Current phase

**Phase:** Bootstrap — **DONE**. Scaffold runs end-to-end. Next phase: first canon trainer (`B-ppo-cot-vizdoom-basic-2B`).

**Iter 3 evidence.** `uv sync --extra dev` resolved 222 packages and installed cleanly (`transformers` pinned to `a29df2d`, `torch 2.6.0+cu124`, `vllm 0.8.5.post1`, 8 × RTX A6000 visible). `flash-attn 2.7.4.post1` built against CUDA 12.5 (cvmfs). `ruff check .` / `ruff format --check .` / `pyright src/cleanrl_vlm` / `mkdocs build --strict` all clean. Pre-commit installed and green. `tests/smoke/test_hello_vlm.py::test_hello_vlm_loads_and_generates` — **PASSED in 92.93s** on `Qwen/Qwen3-VL-2B-Instruct` (per 2026-04-20 backbone-names amendment).

Iter-2 execution of the pre-existing bootstrap plan
[`docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`](docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md):

| Bootstrap Task | Iter status |
|---|---|
| 1 preserve `old`        | iter 1 (branch created; orphan surgery skipped) |
| 2 `.gitignore`          | iter 1 (patched in place) |
| 3 `LICENSE`             | iter 2 |
| 4 `pyproject.toml`      | iter 2 |
| 5 UV init (lockfile + sync) | iter 3 (uv.lock 222 packages; sync green; hatch `allow-direct-references=true` fix) |
| 6 directory skeleton    | iter 2 |
| 7 `CLAUDE.md` rewrite   | iter 2 |
| 8 `README.md`           | iter 2 |
| 9 `CHANGELOG.md`        | iter 1 |
| 10 `LOOP_STATE.md`      | iter 1 (this file) |
| 11 `AUTONOMY_LOG.md`    | iter 1 |
| 12 `MEMORY.md`          | iter 2 |
| 13 `tests/test_imports.py` | iter 2 (12 pass under conda python) |
| 14 `tests/test_spec_exists.py` | iter 2 (4 pass under conda python) |
| 15 `tests/smoke/test_hello_vlm.py` | iter 3 — **PASSED** on `Qwen/Qwen3-VL-2B-Instruct` (AutoModelForImageTextToText, not AutoModelForCausalLM) |
| 16 `.pre-commit-config.yaml` | iter 2 |
| 17 `.github/workflows/ci.yml` | iter 2 |
| 18 `.github/workflows/docs.yml` | iter 2 |
| 19 `mkdocs.yml` + `docs/index.md` | iter 2 (mkdocs build verification deferred to iter 3) |
| 20 doc stubs            | iter 2 |
| 21 full lint + test suite | iter 3 — ruff / ruff-format / pyright / mkdocs-strict / 12 unit + 1 smoke all green |
| 22 single bootstrap commit | **n/a** (adapted to per-task commits in iter 1) |
| 23 scaffold smoke run   | iter 3 — hello-VLM 92.93s on 2B backbone |
| 24 handoff check        | **complete** — /loop is already running; §13 verified in CLAUDE.md by `test_claude_md_carries_autonomy_section` |

## Next task (iter 4)

**ID:** `B-ppo-cot-vizdoom-basic-2B` (renamed from `...-0.8B` per 2026-04-20 backbone-names amendment).

**Objective.** First canon trainer end-to-end: `algos/ppo_cot.py` on `VizdoomBasic-v0` with `Qwen/Qwen3-VL-2B-Instruct`. All applicable §8 invariants passing (Inv-1/3/4/5/6/9/10/11/13/14/15 — the ones that apply to a single-rank first trainer; Inv-7/12/2 come online once checkpointing + long runs land). Tier-1 smoke green. A learning curve the agent judges "genuinely learning" per §0 / §9.

**Procedure (master-spec §13.3).** Orient → brainstorm Plan B (adapted self-approve) → writing-plans → subagent-driven-development → verify → code-reviewer subagent → simplify → commit + journals → dashboard refresh (n/a on first run) → schedule next.

**Sub-decomposition (expected to span multiple /loop iterations).**
1. Brainstorm + amendment (if any) + Plan B file at `docs/superpowers/specs/plans/2026-04-??-ppo-cot-vizdoom-basic.md`.
2. Env layer: port `make_vizdoom_env`, `DiscreteMultiBinaryWrapper`, `ScreenOnlyWrapper`, `action_maps` entries from the prototype `src/utils/` into `src/cleanrl_vlm/envs/` with tests.
3. Model layer: port `BaseVLM` + `CriticHead` + LoRA topology helpers into `src/cleanrl_vlm/models/`, dual-adapter ("actor"/"critic") wired.
4. Prompts: port `prompts/corridor/` / `defend_the_line/` into `src/cleanrl_vlm/prompts/templates/vizdoom/` + prompt builder.
5. Rollout: in-process HF generation path first (spec §3 "Non-generation interfaces skip vLLM" rule applies here — but COT needs generation, so in-process first; vLLM upgrade is `E-vllm-rollout-path`).
6. `algos/ppo_cot.py`: single-file trainer.
7. Inv-1/3/5/9/10/11/13 tests under `tests/invariants/` wired to the new trainer.
8. Inv-15 ground-truth vision probe: `scripts/probe_vision.py` + `prompts/vizdoom/vision_probe.txt` + `docs/vision_probes/vizdoombasic_qwen3-vl-2b/report.md`.
9. Run short training (agent judges length); record CSV + artifacts; iterate per §0 until "genuinely learning" by agent judgment.
10. Code-reviewer subagent on the diff; simplify; commit; journals; pivot to `C-envs-tier1-expand` or `D-invariants-runtime`.

**Constraint.** Long. Expected to consume many /loop iterations. Break into sub-tasks with intermediate commits; each /loop iter picks up at the next incomplete sub-task in the plan file.

## Prioritized task queue (carried from pre-existing plan)

| Priority | Task ID | Objective | Depends on | Status |
|---|---|---|---|---|
| — | `A2-bootstrap-finalize` | UV env + lockfile + smoke | — | **DONE iter 3** |
| 1 | `B-ppo-cot-vizdoom-basic-2B` | First canon trainer end-to-end | — |
| 2 | `C-envs-tier1-expand` | `ALE/Pong-v5` + `MiniGrid-Empty-5x5-v0` as Tier-1 envs | B | pending |
| 3 | `D-invariants-runtime` | Wire `InvariantMonitor` into trainer loop | B | pending |
| 4 | `E-vllm-rollout-path` | Swap in-process generation for vLLM-served COT rollouts | B | pending |
| 5 | `F-canon-expand` | Remaining 8 canon trainers | B | pending |
| 6 | `G-backbone-4B` | Onboard `Qwen/Qwen3-VL-4B-Instruct` per §11 S-6 | B, C | pending |
| 7 | `H-baselines` | cnn_ppo, zero_shot_vlm, frozen_vlm_head | B, C | pending |
| 8 | `I-envs-tier2-full` | All remaining ViZDoom / Atari / Minigrid envs | B, C, F | pending |
| 9 | `J-checkpoint-resume-e2e` | E2E checkpoint/resume + SIGTERM + wandb resume | B, D | pending |
| 10 | `K-research-longhorizon` | First experimental method (e.g., asymmetric VLM critic) | F, I | pending |
| 11 | `L-dashboard` | `scripts/build_dashboard.py` → `docs/RESULTS.md` | B, C, D, F, H | pending |
| 12 | `M-docs-first-milestone` | `scripts/doc_audit.py` on all-green; tag `v0.1.0` | all + all-green | pending |

## Per-combo status

| env | algo | interface | backbone | seeds | status | last_run_id | notes |
|---|---|---|---|---|---|---|---|

(empty — filled by /loop as runs complete)

## Active research threads

None yet.

## Parked / inactive

None yet.

## Leftover pre-scaffold files

The orphan surgery was skipped in iter 1 (safety-floor rationale). The
following files/dirs from the previous prototype remain in the working
tree. They are not imported by `cleanrl_vlm` — safe to leave or prune
later with targeted `git rm`:

- `src/` prototype trainers: `src/models/`, `src/utils/`, `src/train_decoupled_actor_critic_*.py`, `src/ppo.py`, `src/eval.py`, `src/env_debug.py`, `src/tests/`.
- `deepspeed_zero2.yaml`, `deepspeed_zero3.yaml` — superseded by per-run accelerate configs iter 5+.
- `requirements.txt`, `setup.py` — superseded by `pyproject.toml`.
- `test.py` — superseded by `tests/` suite.
- `prompts/<env>/*.txt` — will be mined for content when the new prompt-builder lands.
- `cleanrl-master (1).zip` — gitignored reference dump.

Prune in an iteration after the ported library replaces each one.
