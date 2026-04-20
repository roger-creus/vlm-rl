# LOOP_STATE.md

**Last updated:** 2026-04-20 (iter 2 — scaffold skeleton complete; UV env deferred).
**Maintained by:** the autonomous `/loop` agent; humans read only.

## Current phase

**Phase:** Bootstrap scaffold — **skeleton complete**; env setup + smoke-run deferred to iter 3 (`A2-bootstrap-finalize`).

Iter-2 execution of the pre-existing bootstrap plan
[`docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`](docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md):

| Bootstrap Task | Iter status |
|---|---|
| 1 preserve `old`        | iter 1 (branch created; orphan surgery skipped) |
| 2 `.gitignore`          | iter 1 (patched in place) |
| 3 `LICENSE`             | iter 2 |
| 4 `pyproject.toml`      | iter 2 |
| 5 UV init (lockfile + sync) | **deferred → iter 3 A2** |
| 6 directory skeleton    | iter 2 |
| 7 `CLAUDE.md` rewrite   | iter 2 |
| 8 `README.md`           | iter 2 |
| 9 `CHANGELOG.md`        | iter 1 |
| 10 `LOOP_STATE.md`      | iter 1 (this file) |
| 11 `AUTONOMY_LOG.md`    | iter 1 |
| 12 `MEMORY.md`          | iter 2 |
| 13 `tests/test_imports.py` | iter 2 (12 pass under conda python) |
| 14 `tests/test_spec_exists.py` | iter 2 (4 pass under conda python) |
| 15 `tests/smoke/test_hello_vlm.py` | iter 2 (file committed; run deferred to iter 3 — needs GPU + full deps) |
| 16 `.pre-commit-config.yaml` | iter 2 |
| 17 `.github/workflows/ci.yml` | iter 2 |
| 18 `.github/workflows/docs.yml` | iter 2 |
| 19 `mkdocs.yml` + `docs/index.md` | iter 2 (mkdocs build verification deferred to iter 3) |
| 20 doc stubs            | iter 2 |
| 21 full lint + test suite | **deferred → iter 3 A2** |
| 22 single bootstrap commit | **not applicable** (adapted to per-task commits in iter 1) |
| 23 scaffold smoke run   | **deferred → iter 3 A2** |
| 24 handoff check        | **complete** — /loop is already running; §13 verified in CLAUDE.md by `test_claude_md_carries_autonomy_section` |

## Next task (iter 3)

**ID:** `A2-bootstrap-finalize`.

**Objective.** Make the scaffold runnable end-to-end on a GPU node.

1. `uv venv --python 3.10` + `uv lock` + `uv sync --extra dev` + `uv pip install flash-attn==2.7.4.post1 --no-build-isolation` per bootstrap plan Task 5. Commit `uv.lock`.
2. `uv run pre-commit install` + `uv run pre-commit run --all-files`; fix anything that doesn't pass.
3. Full lint + test suite per bootstrap plan Task 21:
   - `uv run ruff check .` (fix auto-fixable findings; commit fixes).
   - `uv run ruff format --check .` (format if needed; commit).
   - `uv run pyright src/cleanrl_vlm` (warn-only per CI config).
   - `uv run pytest tests/test_imports.py tests/test_spec_exists.py -v`.
   - `uv run mkdocs build --strict`.
4. GPU-dependent hello-VLM smoke per bootstrap plan Task 23:
   - `uv run pytest -m tier1 -v` (expects the GPU + backbone download).
5. Update CHANGELOG / AUTONOMY_LOG / LOOP_STATE; pivot next-task to `B-ppo-cot-vizdoom-basic-0.8B`.

**Constraint.** UV operations resolve + download a few GB of deps; a single iteration may exhaust runtime — break into subiterations as needed, logging each attempt to AUTONOMY_LOG. If a dep resolution fails, web-fetch per S-5 before pinning a workaround.

## Next-next task (iter 4+)

**ID:** `B-ppo-cot-vizdoom-basic-0.8B`.

**Objective** per pre-existing plan's Task-10 content: first canon trainer end-to-end on `VizdoomBasic-v0` with `Qwen/Qwen3.5-VL-0.8B`, all applicable §8 invariants passing, Tier-1 CI green, curve the agent judges "genuinely learning" per §0 / §9.

## Prioritized task queue (carried from pre-existing plan)

| Priority | Task ID | Objective | Depends on |
|---|---|---|---|
| 1 | `A2-bootstrap-finalize` | UV env + lockfile + smoke | — |
| 2 | `B-ppo-cot-vizdoom-basic-0.8B` | First canon trainer end-to-end | A2 |
| 3 | `C-envs-tier1-expand` | `ALE/Pong-v5` + `MiniGrid-Empty-5x5-v0` as Tier-1 envs | B |
| 4 | `D-invariants-runtime` | Wire `InvariantMonitor` into trainer loop | B |
| 5 | `E-vllm-rollout-path` | Swap in-process generation for vLLM-served COT rollouts | B |
| 6 | `F-canon-expand` | Remaining 8 canon trainers | B |
| 7 | `G-backbone-4B` | Onboard `Qwen/Qwen3.5-VL-4B` per §11 S-6 | B, C |
| 8 | `H-baselines` | cnn_ppo, zero_shot_vlm, frozen_vlm_head | B, C |
| 9 | `I-envs-tier2-full` | All remaining ViZDoom / Atari / Minigrid envs | B, C, F |
| 10 | `J-checkpoint-resume-e2e` | E2E checkpoint/resume + SIGTERM + wandb resume | B, D |
| 11 | `K-research-longhorizon` | First experimental method (e.g., asymmetric VLM critic) | F, I |
| 12 | `L-dashboard` | `scripts/build_dashboard.py` → `docs/RESULTS.md` | B, C, D, F, H |
| 13 | `M-docs-first-milestone` | `scripts/doc_audit.py` on all-green; tag `v0.1.0` | all + all-green |

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
