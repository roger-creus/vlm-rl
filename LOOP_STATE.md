# LOOP_STATE.md

**Last updated:** 2026-04-20 (iter 4 — PPO-COT design reviewed + revised).
**Maintained by:** the autonomous `/loop` agent; humans read only.

## Current phase

**Phase:** `B-ppo-cot-vizdoom-basic-2B` — design sign-off. **Design spec committed and code-reviewer-approved at `f7b7fe7`.** Next iteration invokes `superpowers:writing-plans` on §10 of the design to emit the bite-sized implementation plan (27 tasks).

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

## Iter 4 — completed sub-step

- Brainstorm (adapted, §13.1 no user gate) + design spec at
  `docs/superpowers/specs/amendments/2026-04-20-ppo-cot-vizdoom-basic-design.md`
  committed at `794e7a9`.
- `superpowers:code-reviewer` subagent reviewed it (3 blockers / 8
  majors / 12 minors).
- Revised spec at `f7b7fe7` — all blockers resolved in-spec; majors
  assigned to §10 plan tasks; minors folded into simplify pass.

## Next task (iter 5)

**ID:** `B-ppo-cot-vizdoom-basic-2B` — step: writing-plans.

**Objective.** Invoke `superpowers:writing-plans` on §10 of the design
spec to emit `docs/superpowers/specs/plans/2026-04-20-ppo-cot-vizdoom-basic.md`
— 27 bite-sized tasks decomposed from the design.

**Scope of that single iteration.** Just the plan file, committed. No
code yet.

## Iters 6+

Execute the plan task-by-task under
`superpowers:subagent-driven-development`. Each /loop iter picks the
next incomplete task from the plan, runs it (possibly via subagent),
verifies, commits. Milestones where `/loop` iter boundaries naturally
land:

1. Configs + env layer (tasks 1-4)
2. Model layer (tasks 5-8) — first invariant tests green
3. Prompts + rollout (tasks 9-12)
4. Training scaffolding (tasks 13-17) — remaining invariant tests green
5. `algos/ppo_cot.py` assembly (task 19)
6. Integration test green (task 20)
7. Vision probe + backbone probe (tasks 21-22)
8. First training run (task 23) — milestone-gated Wakeup cadence
9. Docs updates + simplify + code-review + journals (tasks 24-27)

Task 23 (the actual training run) is the long pole — agent kicks off via
`run_in_background`, arms a `Monitor` on the metrics-CSV tail with an
alternation grep for `Traceback|Error|FAILED|OOM|elapsed_steps=|ep_return_mean=|inv_.*_status=FAIL`,
and schedules fallback wakeups every ~30 min.

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
