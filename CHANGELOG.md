# Changelog

Per §11 S-12: every `/loop` iteration that changes tracked state adds
an entry here — `{ what, why, evidence, invariants-run }`. Newest on top.
One entry per iteration, not per commit within an iteration.

---

## 2026-04-20 — iter 6 — env layer (plan tasks 1-4/27)

**What.** 4 commits on the PPO-COT port — all of the env layer lands.

- `ff0a19b` (approx) — configs: `configs/backbones.yaml` (2B + 4B),
  `configs/targets.yaml` (VizdoomBasic reference), `configs/envs/
  VizdoomBasic-v0.yaml` (frame_stack.n=1 TODO M6, max_episode_steps=
  null M8, processor pixel override B3 → 76800 native).
- Task 2 — `src/cleanrl_vlm/envs/wrappers.py` + `tests/unit/
  test_env_factory.py`: FrameSkipEnv, ScreenOnlyWrapper,
  DiscreteMultiBinaryWrapper (supersedes prototype's DeadlyCorridor
  + DefendTheLine per m2). 5 tests.
- Task 3 — `src/cleanrl_vlm/envs/vizdoom/{__init__, action_tables,
  factories}.py`: 3 ViZDoom scenarios keyed (Basic, Corridor,
  DefendLine); make_vizdoom_env composes the 3 wrappers + record-stats.
  Adds 1 action_tables test.
- Task 4 — `src/cleanrl_vlm/envs/registry.py`: single make_env dispatch
  (Vizdoom prefix → factory); per-idx seed applied; unknown prefix
  raises KeyError. Adds 2 registry tests.

**Why.** Plan tasks 1-4 execute sequentially — configs before env code
because pixel budget + frame_skip read from YAML. Env layer is a
dependency of every later task.

**Evidence.** `uv run --no-env-file pytest tests/unit/test_env_factory.py`
→ 8 passed in 1.63 s.

**Invariants run.** None yet — invariant tests land in later plan
tasks (8 for Inv-1/3; 11 for Inv-10; 13 for Inv-6; 17 for Inv-4/5/9/11/13).

---

## 2026-04-20 — iter 5 — PPO-COT implementation plan (task B — plan step)

**What.** 27-task implementation plan for task `B-ppo-cot-vizdoom-basic-2B`.
1 commit (`dfc3c4d`).

- `docs/superpowers/specs/plans/2026-04-20-ppo-cot-vizdoom-basic.md`
  (3741 lines) — bite-sized TDD tasks with exact file paths, verbatim
  code blocks, conventional-commits messages, self-review checklist.

**Why.** Design spec at `f7b7fe7` needs task-level decomposition before
implementation iters can consume it. Writing-plans skill output is the
source of truth for iters 6-? to execute under subagent-driven-development.

**Evidence.**
- Line count: 3741 (substantial; delegated to general-purpose subagent
  to conserve main-loop context).
- Self-review checklist at plan tail confirms: 20 §3 module units
  mapped, 9 in-scope invariants mapped to test files, 8 reviewer majors
  encoded, 12 reviewer minors landed.
- Grep for `## Task N:` shows 27 top-level tasks matching design §10.

**Invariants run.** None landed yet — plan describes Inv-1/3/4[single-
path]/5/6/9/10/11/13 test files; implementations land iters 6+.

---

## 2026-04-20 — iter 4 — PPO-COT design spec (task B — design step)

**What.** Design spec for the first canon trainer
(`algos/ppo_cot.py` on `VizdoomBasic-v0` with
`Qwen/Qwen3-VL-2B-Instruct`). 2 commits (`794e7a9`, `f7b7fe7`).

- `docs/superpowers/specs/amendments/2026-04-20-ppo-cot-vizdoom-basic-design.md`
  — 14 sections spanning goal, scope, architecture module table,
  PPO-COT loss formulation, GAE, prompts, Inv-1/3/4/5/6/9/10/11/13
  coverage, TDD test strategy, config defaults, §9-mandate logging
  schema, 27-task deliverable sequence, risks, non-goals, reviewer
  findings resolutions, sign-off.

**Why.** First canon trainer needs a locked design before any code
lands. Design is the source of truth the writing-plans skill will
consume next iteration.

**Evidence.**
- Self-reviewed per §13.1 (no user gate in autonomous mode).
- `superpowers:code-reviewer` subagent reviewed `794e7a9` — 3
  blockers / 8 majors / 12 minors.
- All blockers resolved in-spec at `f7b7fe7`; majors assigned to
  named §10 plan tasks; minors folded into simplify pass (§10 task
  25).
- §13 "Reviewer findings + resolutions" section enumerates each
  finding and its fate.

**Invariants run.** None landed yet (next iter writes plan, iter 6+
implements). Design enumerates iter-4 invariant scope:
Inv-1/3/4[single-path]/5/6/9/10/11/13; deferrals: 2/4[vLLM
variant]/7/8/12/14/15 to named follow-up tasks.

---

## 2026-04-20 — iter 3 — A2-bootstrap-finalize (scaffold runs end-to-end)

**What.** Bootstrap completed. 6 commits (`b8dd216..7fb54bd`).

- `uv.lock` — 222 packages resolved (torch 2.6.0+cu124, transformers
  @ `a29df2d`, vllm 0.8.5, deepspeed, peft 0.19.1, vizdoom 1.3.0).
- `pyproject.toml`: `[tool.hatch.metadata] allow-direct-references=true`
  (needed for transformers-from-git), `[tool.ruff] extend-exclude` for
  prototype leftovers.
- `docs/index.md`: inline quickstart (removed `../README.md` link that
  broke mkdocs --strict).
- `.gitignore`: `site/`, `.claude/` added.
- **Amendment**:
  `docs/superpowers/specs/amendments/2026-04-20-backbone-names-correction.md`
  — Qwen3.5-VL does not exist on HF; use `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct`
  and `-Thinking`. Updated `docs/BACKBONES.md` + `tests/smoke/test_hello_vlm.py`.
- `tests/smoke/test_hello_vlm.py`: switched import from
  `AutoModelForCausalLM` → `AutoModelForImageTextToText` (Qwen3-VL
  registers multimodal).
- Pre-commit hygiene auto-fixes on prototype files committed so future
  pre-commits don't re-touch.
- `docs/TROUBLESHOOTING.md`: flash-attn CUDA_HOME + hatch direct-refs
  tips appended.

**Why.** The scaffold needs to run for the /loop to validate each
subsequent feature. Bootstrap finalize delivers the runnable floor.

**Evidence.**
- `uv sync --extra dev` → all 222 packages installed.
- `uv pip install flash-attn==2.7.4.post1 --no-build-isolation` with
  `CUDA_HOME=/cvmfs/.../cuda/12.5.0` → built in 17 s.
- `uv run --no-env-file ruff check .` → All checks passed.
- `uv run --no-env-file ruff format --check .` → 17 files formatted.
- `uv run --no-env-file pyright src/cleanrl_vlm` → 0 errors / 0 warnings.
- `uv run --no-env-file mkdocs build --strict` → built in 3.49 s (no
  strict-mode warnings).
- `uv run --no-env-file pre-commit run --all-files` → all hooks green.
- `uv run --no-env-file pytest tests/test_imports.py tests/test_spec_exists.py -v`
  → 12 passed under the venv (same as conda python).
- `uv run --no-env-file pytest tests/smoke/test_hello_vlm.py -v -m tier1`
  → **1 passed in 92.93 s** on `Qwen/Qwen3-VL-2B-Instruct`; artifact at
  `/tmp/.../hello_vlm_output.txt` (prompt+response). Proved the dep
  graph + vision processor + generation path all work end-to-end.
- 8 × RTX A6000 (48 GB) visible: `torch.cuda.device_count() == 8`.

**Invariants run.** None of Inv-1..Inv-15 yet — those apply to training
surfaces that don't exist. The hello-VLM smoke is the baseline
"software works" floor that iter-4+ invariant tests build on.

---

## 2026-04-20 — iter 2 — scaffold skeleton

**What.** Execute bootstrap-plan Tasks 3–8, 12–20 on top of iter 1's
loop infra. 9 commits land on master (`408684e`..`4a357c3`).

- `LICENSE` (MIT).
- `pyproject.toml` — UV + ruff + pyright + pytest config; full §3
  dep list (torch 2.6.0, transformers-from-git, peft, accelerate,
  deepspeed, vllm, gymnasium + vizdoom + minigrid + ale-py, imaging,
  logging, dev).
- `src/cleanrl_vlm/` importable package with 6 sub-packages (`envs`,
  `models`, `prompts`, `rollout`, `training`, `research`).
- `algos/`, `baselines/`, `experimental/`, `configs/envs/`,
  `scripts/`, `docs/backbone_probes/`, `docs/vision_probes/`,
  `docs/superpowers/specs/amendments/` all with `.gitkeep`.
- `tests/{invariants,unit,integration,smoke,soak}/__init__.py`.
- `CLAUDE.md` — §13 verbatim at top per Appendix B.
- `README.md` — 1-page pitch + UV quickstart + links (replaces
  the prior prototype's conda-setup README).
- `MEMORY.md` — HTML-commented stub.
- `tests/test_imports.py` (9 tests) + `tests/test_spec_exists.py`
  (4 tests) + `tests/smoke/test_hello_vlm.py` (1 GPU-tier1 test).
- `.pre-commit-config.yaml` — ruff + standard hygiene hooks.
- `.github/workflows/ci.yml` — 3-job matrix (lint / tier1-smoke
  disabled / docs build).
- `.github/workflows/docs.yml` — `mkdocs gh-deploy` on push to master.
- `mkdocs.yml` — material theme, strict build, 12-page nav.
- `docs/index.md` + 12 doc stubs (ARCHITECTURE, ALGORITHMS, ENVS,
  BACKBONES, RECIPES, RESULTS, RESEARCH, INVARIANTS, CHECKPOINTING,
  LOGGING, CONTRIBUTING, TROUBLESHOOTING).

**Why.** Bootstrap plan's §2 scaffold needs to land so the /loop's
subsequent work (canon trainers, env onboarding, etc.) has a place
to live per spec §11 rituals.

**Evidence.** 12/12 CPU-safe tests pass (`PYTHONPATH=src python3 -m
pytest tests/test_imports.py tests/test_spec_exists.py`). `pyproject.toml`
parses under python3's tomllib. `cleanrl_vlm` package imports cleanly.

Deferred verifications (iter 3 `A2-bootstrap-finalize`): `uv lock`
+ `uv sync`, `ruff check .`, `ruff format --check .`, `pyright src/
cleanrl_vlm`, `mkdocs build --strict`, `pytest -m tier1 -v` (hello-VLM
GPU smoke).

**Invariants run.** None applicable — no model, env, or training
surface exists yet.

---

## 2026-04-20 — iter 1 — loop bootstrap

**What.**
- Created local branch `old` pointing at pre-scaffold master HEAD `964377d`
  (§1 "Git strategy").
- Added `LOOP_STATE.md`, `AUTONOMY_LOG.md`, `CHANGELOG.md` (this file) —
  the `/loop` operating surface.
- Patched `.gitignore` with Python / UV / runs / zip ignores on top of
  the pre-existing project-specific ignores.
- Tracked `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`
  and `docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`
  (both were untracked at session start).

**Why.**
Bootstrap the `/loop` operating surface before any scaffold code lands.
The journal / state / changelog files are required by §13.3 step 9;
`old` preservation is required by §1. The spec + pre-existing bootstrap
plan must be tracked so future iterations (which re-read both at start)
are reading from the committed copy.

**Evidence.**
N/A — policy-boilerplate commit, no runtime code paths changed. No
invariants apply yet (no model, env, or training surface exists).

**Invariants run.** None applicable.

**Skills invoked.** `superpowers:using-superpowers`, `superpowers:writing-plans`.

**Adaptations to the pre-existing bootstrap plan (full rationale in
AUTONOMY_LOG).**
- Skipped orphan-branch surgery (`git rm -rf .` + `git clean -fdx`) —
  unnecessary given `old` preserves history; too destructive for the
  §13.1 safety floor in autonomous mode.
- Committing per task group instead of the plan's single Task-22
  monolithic commit — better `/loop` observability.
