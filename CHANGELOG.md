# Changelog

Per §11 S-12: every `/loop` iteration that changes tracked state adds
an entry here — `{ what, why, evidence, invariants-run }`. Newest on top.
One entry per iteration, not per commit within an iteration.

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
