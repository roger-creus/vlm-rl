# LOOP_STATE.md

**Last updated:** 2026-04-20 (iter 1 — loop bootstrap)
**Maintained by:** the autonomous `/loop` agent; humans read only.

## Current phase

**Phase:** Bootstrap scaffold (in progress — partial).

The pre-existing comprehensive plan at
[`docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`](docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md)
is the authoritative scaffold contract. Iter 1 already executed the
subset of that plan that is pure loop infrastructure:

| Bootstrap Task | Iter 1 status |
|---|---|
| Task 1 — preserve `old` branch | done (`git branch old master`) — but the orphan-branch surgery in Steps 4–5 was intentionally skipped (see AUTONOMY_LOG for rationale) |
| Task 2 — `.gitignore` | done (patched the existing `.gitignore`; content differs slightly from plan text but covers the same ignores) |
| Task 9 — `CHANGELOG.md` | done (seeded with iter-1 entry) |
| Task 10 — `LOOP_STATE.md` | done (this file; content interim-until-bootstrap-complete) |
| Task 11 — `AUTONOMY_LOG.md` | done (seeded with iter-1 entry) |
| All other Tasks 3–24 | **pending — iter 2+ work** |

## Next task (iter 2)

**ID:** `A-bootstrap-continue` (finish the pre-existing scaffold plan).

**Objective.** Execute the remaining tasks of
`docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`: Tasks
3 (LICENSE), 4 (pyproject.toml), 5 (init UV), 6 (directory skeleton), 7
(CLAUDE.md rewrite per §13.7), 8 (README.md), 12 (MEMORY.md), 13–15
(tests), 16 (pre-commit), 17–18 (CI workflows), 19 (mkdocs.yml +
docs/index.md), 20 (doc stubs), 21 (run lint + tests), 23 (smoke run),
24 (handoff check). Iter 1's AUTONOMY_LOG entry records the adaptation:
per-task commits, not the plan's single monolithic Task-22 commit.

**Constraint.** Every task lands in a distinct commit with a
conventional-commits-style subject so the /loop observability remains
legible. CHANGELOG.md gets an entry per iteration, not per commit.

**Skip.** Do not run Task 1 Steps 4–5 (orphan branch + `git rm -rf .`).
Master is not being emptied — `old` already preserves history; starting
clean-ish with loop infra committed is fine.

## Next-next task (iter 3+, after bootstrap completes)

**ID:** `B-ppo-cot-vizdoom-basic-0.8B`.

**Objective.** Per the pre-existing plan's post-bootstrap target —
implement `algos/ppo_cot.py` as the first canon trainer end-to-end on
`VizdoomBasic-v0` with `Qwen/Qwen3.5-VL-0.8B`, all applicable §8
invariants passing. Full procedure per the pre-existing plan's Task 10
content and master-spec §13.3.

## Prioritized task queue (after A and B — carried from pre-existing plan)

| Priority | Task ID | Objective | Depends on |
|---|---|---|---|
| 1 | `A-bootstrap-continue` | Finish scaffold plan | (none) |
| 2 | `B-ppo-cot-vizdoom-basic-0.8B` | First canon trainer end-to-end | A |
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
