# cleanrl-vlm Bootstrap Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the `cleanrl-vlm` repo to a state where the autonomous `/loop` agent (defined in master-spec §13) can take over and self-extend. Deliverable is a scaffold + one runnable "hello VLM" smoke test + CI green, committed to a fresh `master` branch with the previous master preserved as `old`.

**Architecture:** Python package managed with **UV**. Source layout `src/cleanrl_vlm/` (hybrid library) + `algos/` (single-file trainers, initially empty) + `baselines/` + `experimental/` + `configs/` + `prompts/` + `scripts/` + `tests/` + `docs/`. `CLAUDE.md` carries master-spec §13 verbatim at its top so every Claude Code session started in this repo reads the autonomy contract first.

**Tech Stack:** Python 3.10+, PyTorch 2.6.0, Transformers (git), PEFT, Accelerate, DeepSpeed, vLLM, Flash-Attn 2.7.4, Gymnasium + ViZDoom + Minigrid + ALE, UV, Ruff, Pyright, Pytest, Rich, W&B, Tensorboard.

**Master spec reference:** `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md` — this plan implements the "scaffold" scope; all subsequent work is the `/loop` agent's responsibility per spec §13.3.

**Pre-conditions:**
- Working directory is the repo root (absolute path: `/network/scratch/r/roger.creus-castanyer/clean-llm-rl`).
- Current `master` is the old prototype (it contains `src/train_decoupled_actor_critic_*.py`, `deepspeed_zero*.yaml`, etc.).
- User has pre-authorized moving the current `master` → branch `old` (master-spec §1, "Git strategy").
- Git user is configured (`user.name` + `user.email`).

---

## Task 1 — Preserve current master as branch `old`

**Files:**
- No new files; git operations only.

Rationale: the spec's git strategy requires preserving existing work on a frozen branch before starting fresh on `master`. User has pre-authorized.

- [ ] **Step 1: Verify clean working tree**

Run:
```bash
git status --short
```
Expected: empty output (no modifications, no untracked files that aren't already part of the existing repo). If there are untracked files, add them to `.gitignore` as appropriate OR commit them on current master first. Do NOT proceed with a dirty tree.

- [ ] **Step 2: Verify we're on master and confirm current HEAD**

Run:
```bash
git branch --show-current
git log -1 --oneline
```
Expected: output line 1 = `master`. Output line 2 = the current tip commit (at time of writing: `964377d generation prompt added`). Record this SHA — it is the tip of `old` after the move.

- [ ] **Step 3: Create branch `old` at current master tip**

Run:
```bash
git branch old
```
Expected: no output (success). Verify with:
```bash
git branch --list old
```
Expected: ` old` listed.

- [ ] **Step 4: Reset master to an empty tree for the new scaffold**

Rationale: rather than destructive `git reset --hard` (which loses history), we create a new empty commit on master and leave `old` pointing at the preserved history. Pure additive, fully reversible.

Run:
```bash
git checkout --orphan scaffold-bootstrap
git rm -rf . 2>/dev/null || true
```
Expected: all files removed from the index. Working tree now empty except for `.git/`.

Then:
```bash
git clean -fdx -e .claude -e docs
```
Expected: working tree cleaned. We keep `.claude/` (local IDE settings) and `docs/` (contains the committed spec) by exception.

- [ ] **Step 5: Verify preservation**

Run:
```bash
git log old --oneline -5
```
Expected: shows the preserved history of the old master (including `964377d`, `7887ce8`, `cb02ec9`, etc.). Confirms `old` is intact.

Then:
```bash
ls -la
```
Expected: working directory contains `.git/`, `.claude/`, `docs/` only.

- [ ] **Step 6: Restore `docs/superpowers/specs/` tracking on the new orphan branch**

Rationale: the spec + this plan already exist on disk under `docs/superpowers/specs/`. They were authored before the git reset and need to be re-added to the new branch's index.

Run:
```bash
git add docs/superpowers/specs/
git status --short
```
Expected: `A docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md` and `A docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`.

- [ ] **Step 7: Do NOT commit yet** — subsequent tasks will add more scaffold files to this same initial commit.

---

## Task 2 — `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore` with all standard Python + ML ignores**

Write to `.gitignore`:

```gitignore
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual envs
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# UV
.uv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
.ipynb_checkpoints/

# Testing / coverage
.coverage
.coverage.*
.cache
.pytest_cache/
.tox/
.nox/
htmlcov/
coverage.xml
*.cover
*.py,cover
.hypothesis/

# Type-checking
.mypy_cache/
.dmypy.json
.pyre/
.pytype/

# Logs / runs
runs/
wandb/
*.log
lightning_logs/
tb_logs/

# Model / data artifacts
*.pt
*.ckpt
*.safetensors
*.bin
hub/
.cache/

# OS
.DS_Store
Thumbs.db

# Local Claude Code settings
.claude/settings.local.json

# Repo-specific — old prototype zip
cleanrl-master*.zip
```

- [ ] **Step 2: Stage**

Run:
```bash
git add .gitignore
```

---

## Task 3 — `LICENSE` (MIT)

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write the MIT LICENSE**

Write to `LICENSE`:

```
MIT License

Copyright (c) 2026 Roger Creus Castanyer and cleanrl-vlm contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add LICENSE
```

---

## Task 4 — `pyproject.toml` with UV + deps

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

Write to `pyproject.toml`:

```toml
[project]
name = "cleanrl-vlm"
version = "0.0.1"
description = "CleanRL-style library + research paper for online RL finetuning of Vision-Language Models in interactive visual environments."
authors = [{ name = "Roger Creus Castanyer", email = "roger@vmax.ai" }]
readme = "README.md"
requires-python = ">=3.10,<3.13"
license = { text = "MIT" }
keywords = [
    "reinforcement learning",
    "vision-language models",
    "PPO",
    "GRPO",
    "RLOO",
    "LoRA",
    "VLM",
    "atari",
    "vizdoom",
    "minigrid",
    "research",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Core ML
    "torch==2.6.0",
    "torchvision==0.21.0",
    "torchaudio==2.6.0",
    "transformers @ git+https://github.com/huggingface/transformers",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "deepspeed>=0.15.0",
    "vllm>=0.7.0",
    # Envs
    "gymnasium>=1.0.0",
    "vizdoom>=1.2.3",
    "minigrid>=3.0.0",
    "ale-py>=0.9.0",
    "AutoROM[accept-rom-license]>=0.6.1",
    # Imaging
    "opencv-python>=4.10.0",
    "pillow>=10.4.0",
    # Scientific
    "numpy>=1.26.0,<2",
    "scipy>=1.13.0",
    "pandas>=2.2.0",
    "pyarrow>=17.0.0",
    # Config + CLI
    "pyyaml>=6.0",
    "tyro>=0.8.0",
    # Logging
    "rich>=13.0.0",
    "wandb>=0.18.0",
    "tensorboard>=2.17.0",
    # Utilities
    "tqdm>=4.66.0",
    "huggingface-hub>=0.25.0",
]

[project.optional-dependencies]
perf = [
    # flash-attn requires torch installed first + --no-build-isolation;
    # document separate install step in README.
    # listed here so tooling surfaces it, but UV users install manually:
    #   uv pip install flash-attn==2.7.4.post1 --no-build-isolation
]
dev = [
    "pytest>=8.3.0",
    "pytest-timeout>=2.3.0",
    "pytest-xdist>=3.6.0",
    "ruff>=0.6.0",
    "pyright>=1.1.380",
    "pre-commit>=4.0.0",
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
]

[project.urls]
Homepage = "https://github.com/roger-creus/cleanrl-vlm"
Documentation = "https://roger-creus.github.io/cleanrl-vlm"
Repository = "https://github.com/roger-creus/cleanrl-vlm"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cleanrl_vlm"]

[tool.uv]
# UV handles most installation automatically; flash-attn is still a separate step.

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
ignore = [
    "E501",   # line-too-long (handled by formatter)
    "B008",   # function-call-in-default-argument (common for typer/tyro)
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["E402"]  # late imports in tests are fine
"algos/**" = ["E402"]  # single-file trainers sometimes need late imports

[tool.pyright]
include = ["src/cleanrl_vlm"]
exclude = ["**/__pycache__", "**/.venv", "**/build"]
pythonVersion = "3.10"
typeCheckingMode = "basic"
reportMissingImports = "warning"
reportMissingTypeStubs = false

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]
markers = [
    "tier1: fast smoke tests that run on CI (0.8B backbone, short runs)",
    "tier2: overnight / paper runs (4B backbone, longer runs)",
    "invariant: correctness invariants from master-spec §8",
    "gpu: requires at least one GPU",
    "vllm: requires vLLM to be installed and loadable",
]
timeout = 600
```

- [ ] **Step 2: Verify TOML is parseable**

Run:
```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Stage**

Run:
```bash
git add pyproject.toml
```

---

## Task 5 — Initialize UV project

**Files:**
- Create: `uv.lock` (auto-generated by UV).

- [ ] **Step 1: Ensure UV is installed**

Run:
```bash
which uv || (echo "UV not installed — install from https://docs.astral.sh/uv/ first"; exit 1)
```
Expected: a path to `uv`. If not installed, install via `curl -LsSf https://astral.sh/uv/install.sh | sh` and re-run.

- [ ] **Step 2: Create the UV virtual environment for the project**

Run:
```bash
uv venv --python 3.10
```
Expected: creates `.venv/` with Python 3.10. If Python 3.10 is not on PATH, specify the path (e.g., `uv venv --python /usr/bin/python3.10`).

- [ ] **Step 3: Resolve and lock dependencies**

Run:
```bash
uv lock
```
Expected: `uv.lock` file created; resolution succeeds. If resolution fails for a specific dep, investigate (may need to pin a different version; web-fetch per master-spec S-5 rule).

- [ ] **Step 4: Install deps into the venv**

Run:
```bash
uv sync --extra dev
```
Expected: all core + dev deps installed. Likely takes several minutes on first run (torch + transformers are large).

- [ ] **Step 5: Install flash-attn separately (doc in README for humans)**

Run:
```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```
Expected: builds and installs. Takes 5-10 minutes on first build. If it fails with a CUDA version mismatch, document in `docs/TROUBLESHOOTING.md` (see Task 26) and continue — later smoke tests will catch the issue if attention kernels are broken at runtime.

- [ ] **Step 6: Verify core imports work**

Run:
```bash
uv run python -c "
import torch
import transformers
import peft
import accelerate
import deepspeed
import vllm
import gymnasium
import vizdoom
import minigrid
import ale_py
import rich
import wandb
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('accelerate:', accelerate.__version__)
print('deepspeed:', deepspeed.__version__)
print('vllm:', vllm.__version__)
print('gymnasium:', gymnasium.__version__)
print('ALL IMPORTS OK')
"
```
Expected: version numbers printed for each package, final line `ALL IMPORTS OK`. If any import fails, debug (per S-5 web-fetch the lib's docs first) before continuing.

- [ ] **Step 7: Stage `uv.lock`**

Run:
```bash
git add uv.lock
```

---

## Task 6 — Directory skeleton

**Files:** (all directories + placeholder `__init__.py` or `.gitkeep` files)

- Create: `src/cleanrl_vlm/__init__.py`
- Create: `src/cleanrl_vlm/envs/__init__.py`
- Create: `src/cleanrl_vlm/models/__init__.py`
- Create: `src/cleanrl_vlm/prompts/__init__.py`
- Create: `src/cleanrl_vlm/rollout/__init__.py`
- Create: `src/cleanrl_vlm/training/__init__.py`
- Create: `src/cleanrl_vlm/research/__init__.py`
- Create: `algos/.gitkeep`
- Create: `experimental/.gitkeep`
- Create: `baselines/.gitkeep`
- Create: `configs/envs/.gitkeep`
- Create: `prompts/.gitkeep`
- Create: `scripts/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/invariants/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/smoke/__init__.py`
- Create: `tests/soak/__init__.py`
- Create: `docs/backbone_probes/.gitkeep`
- Create: `docs/vision_probes/.gitkeep`

- [ ] **Step 1: Create all directories**

Run:
```bash
mkdir -p \
  src/cleanrl_vlm/envs \
  src/cleanrl_vlm/models \
  src/cleanrl_vlm/prompts \
  src/cleanrl_vlm/rollout \
  src/cleanrl_vlm/training \
  src/cleanrl_vlm/research \
  algos \
  experimental \
  baselines \
  configs/envs \
  prompts \
  scripts \
  tests/invariants \
  tests/unit \
  tests/integration \
  tests/smoke \
  tests/soak \
  docs/backbone_probes \
  docs/vision_probes
```
Expected: no output; all directories created.

- [ ] **Step 2: Write `src/cleanrl_vlm/__init__.py`**

Content:
```python
"""cleanrl-vlm — online RL finetuning of Vision-Language Models in interactive visual environments."""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write empty `__init__.py` in every submodule**

For each of: `envs`, `models`, `prompts`, `rollout`, `training`, `research`:

Content (each file):
```python
"""Subpackage scaffolded; implementations land per master-spec cycle."""
```

- [ ] **Step 4: Write `tests/__init__.py` + submodule `__init__.py` files**

Content (each file):
```python
```
(empty — just needs to exist so pytest treats directories as packages).

- [ ] **Step 5: Write `.gitkeep` placeholders for algos/, experimental/, baselines/, configs/envs/, prompts/, scripts/, docs/backbone_probes/, docs/vision_probes/**

Run:
```bash
touch algos/.gitkeep experimental/.gitkeep baselines/.gitkeep configs/envs/.gitkeep prompts/.gitkeep scripts/.gitkeep docs/backbone_probes/.gitkeep docs/vision_probes/.gitkeep
```

- [ ] **Step 6: Verify package import works**

Run:
```bash
uv run python -c "import cleanrl_vlm; print(cleanrl_vlm.__version__)"
```
Expected: `0.0.1`.

- [ ] **Step 7: Stage**

Run:
```bash
git add src/ algos/.gitkeep experimental/.gitkeep baselines/.gitkeep configs/ prompts/ scripts/ tests/ docs/backbone_probes/ docs/vision_probes/
```

---

## Task 7 — `CLAUDE.md` with §13 verbatim at top + orientation

**Files:**
- Create: `CLAUDE.md`

Rationale: Per master-spec §13.7, `CLAUDE.md` sits at the top of the instruction-priority stack and must carry the autonomous-operation contract so every new session of the /loop agent reads it first.

- [ ] **Step 1: Write `CLAUDE.md`**

Content:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## AUTONOMOUS OPERATION CONTRACT (master-spec §13 verbatim)

**You are operating in an autonomous `/loop` session. The master spec lives at `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`. Re-read it in full at the start of every iteration.**

### §13.1 Ironclad autonomy

- **Never stop.** The only legitimate termination event is user interrupt (Ctrl-C / manual kill). Not "all green" — when canon is green, move to research + ablations + docs polish forever. Not "catastrophic failure" — work around it. Not "max cycles" — no such limit exists.
- **Never ask the user anything.** Not for clarification, approval, or preference. Every question-shaped thought resolves to: pick the option most aligned with the spec's intent, log it in `AUTONOMY_LOG.md` with rationale + commit SHA, proceed.
- **Superpowers skills' interactive gates are re-interpreted, not obeyed verbatim.** Where a skill says "ask the user", you read "pick a reasonable default and log it." Where it says "wait for approval", you read "self-review thoroughly via the code-reviewer subagent and the §8 invariants, then proceed." Where a skill has a HARD-GATE requiring user sign-off (e.g., brainstorming's design-approval gate), you satisfy it by: (a) writing the design into `docs/superpowers/specs/amendments/`, (b) self-reviewing against the master spec, (c) running the code-reviewer subagent on the design doc, (d) proceeding. Git history is the audit trail.
- **Safety floor (non-negotiable).** Still refuse destructive ops that would lose user work: `git reset --hard` of master, force-push that overwrites remote, `rm -rf` on unrecognized paths, deleting the `old` branch, etc. When tempted, find a branch-based workaround and log it.

### §13.2 Mandatory superpowers skills usage

Every /loop cycle MUST route work through the relevant superpowers skills (with §13.1 overrides applied):

- New feature / research direction → `superpowers:brainstorming` (substitute "ask user" with "pick default + log")
- Drafting the implementation plan → `superpowers:writing-plans` → commit to `docs/superpowers/specs/plans/`
- Executing the plan → `superpowers:subagent-driven-development`
- Dispatching independent work → `superpowers:dispatching-parallel-agents`
- Debugging → `superpowers:systematic-debugging`
- New code with tests → `superpowers:test-driven-development`
- Verifying work is done → `superpowers:verification-before-completion`
- Reviewing finished work → `superpowers:code-reviewer` subagent (via Agent tool)
- Cleaning up → `simplify` skill
- Isolating risky work → `superpowers:using-git-worktrees`
- Finishing a branch → `superpowers:finishing-a-development-branch`

### §13.3 Canonical /loop iteration shape

1. Orient (read spec + LOOP_STATE.md + last 20 AUTONOMY_LOG entries + CHANGELOG tail).
2. Pick next task from prioritized queue.
3. Brainstorm (adapted, self-approved).
4. Plan via writing-plans → commit plan.
5. Execute via subagent-driven-development.
6. Verify via verification-before-completion + §8 invariants.
7. Review via code-reviewer subagent.
8. Simplify via simplify skill.
9. Commit + update `CHANGELOG.md`, `AUTONOMY_LOG.md`, `LOOP_STATE.md`.
10. Refresh dashboard (`docs/RESULTS.md`) if Tier-2 results changed.
11. Schedule next iteration (self-pace).

### §13.4 Anti-patterns ruled out

"Let me ask the user" → NO, pick spec-aligned default + log.
"I'll wait for approval before merging" → NO, self-approve via code-reviewer + invariants + tests.
"Let me just quickly fix without a test" → NO, S-1 requires tests in the same commit.
"This optimization is equivalent so no parity test needed" → NO, S-3 requires Inv-4 parity.
"I'll skip the invariant this time" → NO, §8's point is that silent bugs pass obvious-looking code.
"I should stop when canon goes green" → NO, research + ablations + docs continue forever.
"This threshold says red so red" → NO, §0: investigate → understand → iterate; hard red only for genuine correctness bugs.

### §13.5 Thresholds are signals, not gates (master-spec §0)

Numerical thresholds in the spec (accuracies, tolerances, step counts) are **starting points for your judgment**, not CI-style cliffs. When a threshold fires: investigate, understand, iterate. Hard failures are reserved for genuine correctness bugs (shape mismatches, NaNs, true training↔inference divergence, checkpoint round-trip breakage, non-determinism under fixed seed).

---

## Repo orientation

**Git strategy:** current `master` is the fresh scaffold. The previous prototype is preserved on branch `old` — read-only, never modified.

**Package:** `src/cleanrl_vlm/` (library). `algos/` holds 9 canon trainers (initially empty; populated by /loop). `experimental/` holds research playground. `baselines/` holds CNN-PPO, zero-shot-VLM, frozen-VLM+head baselines.

**Run quickstart (scaffold verification):**
```bash
uv sync --extra dev
uv run pytest -m tier1 -v
```

**Common commands:**
- `uv run pytest`                      — run full test suite
- `uv run pytest -m tier1`             — fast smoke tests only
- `uv run pytest -m invariant`         — correctness invariants
- `uv run ruff check .`                — lint
- `uv run ruff format .`               — format
- `uv run pyright src/cleanrl_vlm`     — type check
- `uv run pre-commit run --all-files`  — pre-commit hooks
- `uv run python scripts/probe_vision.py <env> <backbone>` — Inv-15 vision probe (once implemented)

**Launching the autonomous /loop (human starts a fresh session):**
```
/loop

You are the autonomous /loop agent for cleanrl-vlm. Master spec at
docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md — read it in
full at the start of every iteration. Follow §13 strictly. Use the
superpowers skills continuously. Never ask me questions. Never stop except
on Ctrl-C. All decisions go in AUTONOMY_LOG.md.
```

**Journals (append-only as you work):**
- `LOOP_STATE.md` — next-task pointer, per-combo status table.
- `AUTONOMY_LOG.md` — every non-trivial decision with rationale + SHA.
- `CHANGELOG.md` — every PR's { what, why, evidence, invariants-run }.
- `docs/RESULTS.md` — live dashboard (auto-generated from metric CSVs).
- `docs/RESEARCH.md` — research journal (auto-appended).

**Hardware available:** 1 node × 8 × NVIDIA RTX A6000 (48 GB each). See master-spec §1 for precision + batch-floor constraints.

---

## Memory system

You have access to a persistent memory system. See your system prompt for usage; update it as you learn things about the codebase, the user's preferences, or ongoing work.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add CLAUDE.md
```

---

## Task 8 — `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Content:

```markdown
# cleanrl-vlm

**CleanRL-style library + research paper for online RL finetuning of Vision-Language Models in interactive visual environments (ViZDoom, Atari, Minigrid).**

The promise: pretrained VLMs + efficient LoRA finetuning + a correct on-policy RL pipeline (PPO / GRPO / RLOO) can adapt a static foundation model into an agentic one that outperforms CNN agents trained from scratch — while training a small fraction of parameters. Along the way, we contribute novel methods for long-horizon credit assignment with VLM critics.

## Quickstart

```bash
# 1. Install UV: https://docs.astral.sh/uv/

# 2. Set up the environment
uv venv --python 3.10
uv sync --extra dev

# 3. Install flash-attn (separate because of build-isolation requirement)
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# 4. Run the smoke test
uv run pytest -m tier1 -v
```

## Canonical trainers (spec §5 — initially empty; populated by /loop)

`algos/{ppo, grpo, rloo}_{cot, action, head}.py` — 9 single-file trainers. COT is the hero interface; action-scoring and MLP-head are ablations.

## Baselines (spec §7)

- `baselines/cnn_ppo.py` — from-scratch CNN PPO (non-VLM)
- `baselines/zero_shot_vlm.py` — pure prompting, no RL
- `baselines/frozen_vlm_head.py` — frozen VLM + trainable MLP head (no LoRA)

## Documentation

See `docs/index.md` (mkdocs-rendered). Key pages:

- `docs/ARCHITECTURE.md` — library layout + data flow
- `docs/ALGORITHMS.md` — per-algo math + code pointers
- `docs/ENVS.md` — env catalogue + target scores
- `docs/BACKBONES.md` — supported VLMs
- `docs/RECIPES.md` — copy-pasteable reproduction commands
- `docs/RESULTS.md` — live benchmark dashboard
- `docs/RESEARCH.md` — research journal
- `docs/INVARIANTS.md` — correctness invariants Inv-1..Inv-15
- `docs/CONTRIBUTING.md` — onboarding rituals for new envs/algos/backbones
- `docs/TROUBLESHOOTING.md` — OOM, NaN, logprob drift, vLLM issues

**Master spec (the single source of truth for design):** `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`.

## License

MIT. See `LICENSE`.

## Citation

To be populated on first tagged release (`v0.1.0`, triggered when all Tier-1 + hero Tier-2 combos land green simultaneously per spec §12).

## Previous prototype

The prior iteration of this project is preserved on branch `old` for reference. It is frozen and no longer developed.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add README.md
```

---

## Task 9 — `CHANGELOG.md`

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `CHANGELOG.md`**

Content:

```markdown
# Changelog

All notable changes to cleanrl-vlm land here. Format per master-spec §11 S-12: each entry is `{ what, why, evidence, invariants-run }`.

The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.0.1] — 2026-04-19 — Bootstrap scaffold

- **What.** Fresh `master` branch created from scratch; previous prototype preserved on branch `old`. Bootstrap scaffold: `pyproject.toml` + UV + directory skeleton + `CLAUDE.md` with master-spec §13 autonomy contract + `LICENSE` (MIT) + `.gitignore` + first "hello VLM" smoke test that loads `Qwen/Qwen3.5-VL-0.8B` and generates one token. CI wired (ruff + pyright + pytest + docs build). Docs stubs present.
- **Why.** Master spec `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md` (committed with this entry) requires a fresh scaffold before the autonomous `/loop` agent takes over per §13.
- **Evidence.** `uv run pytest -m tier1 -v` passes locally + on CI. `uv run ruff check .` clean. `uv run pyright src/cleanrl_vlm` clean. `uv run mkdocs build --strict` succeeds.
- **Invariants run.** None applicable yet (Inv-1..Inv-15 apply to trainer runs; scaffold has no trainer). The "hello VLM" smoke verifies only that the model loads + generates.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add CHANGELOG.md
```

---

## Task 10 — `LOOP_STATE.md`

**Files:**
- Create: `LOOP_STATE.md`

- [ ] **Step 1: Write `LOOP_STATE.md`**

Content:

```markdown
# LOOP_STATE.md

**Last updated:** 2026-04-19 (bootstrap scaffold committed)
**Maintained by:** the autonomous /loop agent; humans read only.

## Next task

**ID:** `B-ppo-cot-vizdoom-basic-0.8B`

**Objective.** Generate and execute Plan B: implement `algos/ppo_cot.py` (PPO with the COT action interface, master-spec §5) as the first canon trainer. Target: train for an agent-chosen number of iterations on `VizdoomBasic-v0` with backbone `Qwen/Qwen3.5-VL-0.8B`, all applicable §8 invariants passing (Inv-1..Inv-15 including the ground-truth vision probe), Tier-1 smoke green on CI, a curve of episodic return over time that the agent judges as "genuinely learning" per §0 / §9.

**Procedure.** Apply the canonical /loop iteration shape (master-spec §13.3):

1. Orient: read master spec in full; read `CHANGELOG.md` tail; read this file.
2. Brainstorm (adapted): use `superpowers:brainstorming` to design Plan B; self-approve via §13.1 substitution; write an amendment to `docs/superpowers/specs/amendments/YYYY-MM-DD-ppo-cot-plan-b.md` only if it extends the master spec.
3. Plan: use `superpowers:writing-plans` → commit `docs/superpowers/specs/plans/YYYY-MM-DD-ppo-cot-vizdoom-basic.md`.
4. Execute: use `superpowers:subagent-driven-development` to work task-by-task. Parallelize independent tasks.
5. Verify: run Inv-1..Inv-15 where applicable; run the smoke test; confirm §9 evidence gates by the agent's own judgment.
6. Review: launch `superpowers:code-reviewer` subagent on the diff; fix findings.
7. Simplify: run `simplify` skill on the changed files.
8. Commit + journals: update `CHANGELOG.md`, `AUTONOMY_LOG.md`, this file.
9. Refresh dashboard: regenerate `docs/RESULTS.md` if Tier-2 results changed (not applicable for this first task on 0.8B / Tier-1 smoke).
10. Schedule next iteration.

## Prioritized task queue (loose — agent judges order per §13.3)

| Priority | Task ID | Objective | Depends on |
|----------|---------|-----------|------------|
| 1 | `B-ppo-cot-vizdoom-basic-0.8B` | First canon trainer end-to-end on Tier-1 smoke env | (none — bootstrap done) |
| 2 | `C-envs-tier1-expand` | Add `ALE/Pong-v5` + `MiniGrid-Empty-5x5-v0` as Tier-1 envs with full onboarding ritual (spec §11 S-7) | B |
| 3 | `D-invariants-runtime` | Wire `InvariantMonitor` into the trainer loop so Inv-1..Inv-15 sample at runtime (not only at CI) | B |
| 4 | `E-vllm-rollout-path` | Swap the initial in-process generation path for vLLM-served COT rollouts (spec §3 rollout subsection) | B |
| 5 | `F-canon-expand` | Implement remaining 8 canon trainers (PPO × {action, head}, GRPO × {cot, action, head}, RLOO × {cot, action, head}) | B |
| 6 | `G-backbone-4B` | Onboard `Qwen/Qwen3.5-VL-4B` per spec §11 S-6 | B, C |
| 7 | `H-baselines` | Implement cnn_ppo, zero_shot_vlm, frozen_vlm_head baselines | B, C |
| 8 | `I-envs-tier2-full` | Add all remaining ViZDoom / Atari / Minigrid envs per §4 | B, C, F |
| 9 | `J-checkpoint-resume-e2e` | End-to-end checkpoint/resume test including SIGTERM + env state + wandb resume (spec §10) | B, D |
| 10 | `K-research-longhorizon` | First experimental method from spec §5 (e.g., asymmetric VLM critic) | F, I |
| 11 | `L-dashboard` | Implement `scripts/build_dashboard.py` → `docs/RESULTS.md` auto-generation | B, C, D, F, H |
| 12 | `M-docs-first-milestone` | Run `scripts/doc_audit.py` on first all-green milestone; tag v0.1.0 | all above + all-green |

## Per-combo status (empty; filled by /loop as runs complete)

| env | algo | interface | backbone | seeds | status | last_run_id | notes |
|-----|------|-----------|----------|-------|--------|-------------|-------|

## Active research threads

None yet.

## Parked / inactive

None yet.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add LOOP_STATE.md
```

---

## Task 11 — `AUTONOMY_LOG.md`

**Files:**
- Create: `AUTONOMY_LOG.md`

- [ ] **Step 1: Write `AUTONOMY_LOG.md`**

Content:

```markdown
# AUTONOMY_LOG.md

Append-only journal of every non-trivial decision the autonomous `/loop` agent makes. Entries include timestamp, rationale, and commit SHA. Newest entries appended at the bottom.

Format:

```
### YYYY-MM-DD HH:MM — <decision-slug> — @<commit-sha>

**Context.** What was the situation.
**Options considered.** Multiple choice + trade-offs.
**Choice.** The one picked.
**Rationale.** Why, with reference to master spec sections where relevant.
**Follow-ups.** Anything to revisit.
```

---

### 2026-04-19 — bootstrap-scaffold — @(pending commit SHA)

**Context.** Initial scaffold of the `cleanrl-vlm` repo. User preserved prior prototype as branch `old`; new `master` is empty and bootstrapped via Plan A (`docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`).

**Options considered.** (a) Bootstrap minimum + first trainer in one plan (~100 tasks). (b) Bootstrap minimum only; /loop writes subsequent plans itself. (c) Pre-write 10 plans.

**Choice.** (b). Bootstrap scaffold only.

**Rationale.** Master-spec §13.2 mandates that the /loop agent use `superpowers:writing-plans` for every substantial feature. Writing detailed plans ahead of time would pre-commit the agent's path; it's more faithful to the autonomy contract to hand off at the earliest point that enables productive /loop iteration.

**Follow-ups.** /loop's first task (`B-ppo-cot-vizdoom-basic-0.8B`, see `LOOP_STATE.md`) is to write Plan B.
```

- [ ] **Step 2: Stage**

Run:
```bash
git add AUTONOMY_LOG.md
```

---

## Task 12 — `MEMORY.md` scaffolding

**Files:**
- Create: `MEMORY.md`

Rationale: some Claude Code harnesses use `MEMORY.md` as an index into a memory system. Pre-create it as an empty index so subsequent /loop sessions don't waste cycles on "does this exist?"

- [ ] **Step 1: Write `MEMORY.md`**

Content:

```markdown
<!--
Index into the memory system. One line per memory file under
.claude/.../memory/, format: `- [Title](file.md) — one-line hook`.
See the system-level memory-skill docs for canonical format.
-->
```

- [ ] **Step 2: Stage**

Run:
```bash
git add MEMORY.md
```

---

## Task 13 — `tests/test_imports.py` (TDD — import-level sanity)

**Files:**
- Create: `tests/test_imports.py`

- [ ] **Step 1: Write the test**

Content:

```python
"""Verify that the scaffolded package and its submodules import cleanly.

Follows master-spec §0: binary correctness (either imports work or they don't).
"""

import importlib

import pytest


SUBMODULES = [
    "cleanrl_vlm",
    "cleanrl_vlm.envs",
    "cleanrl_vlm.models",
    "cleanrl_vlm.prompts",
    "cleanrl_vlm.rollout",
    "cleanrl_vlm.training",
    "cleanrl_vlm.research",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name: str) -> None:
    mod = importlib.import_module(name)
    assert mod is not None, f"importlib returned None for {name!r}"


def test_version_string() -> None:
    import cleanrl_vlm

    assert isinstance(cleanrl_vlm.__version__, str)
    assert cleanrl_vlm.__version__ == "0.0.1"
```

- [ ] **Step 2: Run the test**

Run:
```bash
uv run pytest tests/test_imports.py -v
```
Expected: all tests PASS (8 parametrized + 1 version = 9 tests passed).

- [ ] **Step 3: Stage**

Run:
```bash
git add tests/test_imports.py
```

---

## Task 14 — `tests/test_spec_exists.py` (sanity that the spec is present)

**Files:**
- Create: `tests/test_spec_exists.py`

- [ ] **Step 1: Write the test**

Content:

```python
"""Verify the master spec file exists and is non-empty.

Motivation: the spec is the single source of truth for the /loop agent; if
it goes missing, every subsequent cycle is compromised. This test is a
cheap tripwire.
"""

from pathlib import Path


SPEC_PATH = Path("docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md")


def test_spec_exists() -> None:
    assert SPEC_PATH.exists(), f"master spec missing at {SPEC_PATH}"


def test_spec_non_empty() -> None:
    assert SPEC_PATH.stat().st_size > 1000, (
        f"master spec suspiciously small ({SPEC_PATH.stat().st_size} bytes); "
        "expected > 1000"
    )


def test_spec_has_autonomy_section() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")
    assert "## §13. Autonomous operation contract" in content
    assert "Never stop" in content
    assert "Never ask the user anything" in content


def test_claude_md_carries_autonomy_section() -> None:
    claude_md = Path("CLAUDE.md")
    assert claude_md.exists(), "CLAUDE.md missing"
    content = claude_md.read_text(encoding="utf-8")
    assert "AUTONOMOUS OPERATION CONTRACT" in content
    assert "Never stop" in content
    assert "Never ask the user anything" in content
```

- [ ] **Step 2: Run the test**

Run:
```bash
uv run pytest tests/test_spec_exists.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 3: Stage**

Run:
```bash
git add tests/test_spec_exists.py
```

---

## Task 15 — `tests/smoke/test_hello_vlm.py` — end-to-end backbone load + generate

**Files:**
- Create: `tests/smoke/test_hello_vlm.py`

Rationale: this is the scaffold's "software is working" floor. Proves deps install, Qwen3.5-VL-0.8B loads, processor handles images, one-shot generation works.

- [ ] **Step 1: Write the test**

Content:

```python
"""Hello-VLM smoke: load Qwen/Qwen3.5-VL-0.8B, feed a synthetic image + prompt, generate a response.

Verifies:
  * Dependency graph resolves and imports.
  * Backbone loads (requires transformers from git per master-spec §3).
  * AutoProcessor handles multimodal inputs.
  * Generation path does not error.

This is NOT a correctness test for learning — it only proves the scaffold
infra is runnable. Master-spec §0 governs the response to any failure:
investigate → understand → iterate. Hard-fail only on genuine correctness
bugs (wrong shapes, NaN, import errors).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


MODEL_ID = os.environ.get("CLEANRL_VLM_SMOKE_MODEL", "Qwen/Qwen3.5-VL-0.8B")


@pytest.fixture(scope="module")
def synthetic_image() -> Image.Image:
    """A 256x256 RGB image with four distinct colored quadrants + a text label overlay."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # top-left red, top-right green, bottom-left blue, bottom-right yellow
    arr[:128, :128] = [220, 20, 20]
    arr[:128, 128:] = [20, 200, 20]
    arr[128:, :128] = [20, 20, 220]
    arr[128:, 128:] = [230, 230, 20]
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.tier1
@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_hello_vlm_loads_and_generates(synthetic_image: Image.Image, tmp_path: Path) -> None:
    """Load Qwen3.5-VL-0.8B, feed the synthetic quadrant image + 'Describe the image.' prompt, generate."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoProcessor

    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "List the dominant colors in each quadrant of this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[synthetic_image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    decoded = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Artifact: dump the prompt + response under tmp_path for inspection on CI failure.
    (tmp_path / "hello_vlm_output.txt").write_text(
        f"PROMPT:\n{text}\n\nRESPONSE:\n{decoded}\n", encoding="utf-8"
    )

    # The ONLY assertion is that generation produced at least one token of output.
    # We do not assert the content (that's the ground-truth vision probe's job — Inv-15).
    assert decoded.strip(), "model produced empty output"
    assert len(generated[0]) > inputs.input_ids.shape[1], "no new tokens generated"
```

- [ ] **Step 2: Run the test locally**

Run:
```bash
uv run pytest tests/smoke/test_hello_vlm.py -v -m tier1
```
Expected:
- On a machine with a CUDA GPU + transformers-from-git installed: PASS (takes ~1-2 minutes first time as the model downloads).
- On a CPU-only machine: SKIPPED with reason "smoke test requires CUDA".

If it fails with an import error for `AutoProcessor` on Qwen3.5-VL-0.8B, web-fetch the latest transformers docs (per master-spec S-5) to confirm the API; update the test. If the model fails to load, web-fetch the Qwen3.5-VL-0.8B model card for the current recommended loading pattern.

- [ ] **Step 3: Stage**

Run:
```bash
git add tests/smoke/test_hello_vlm.py
```

---

## Task 16 — `.pre-commit-config.yaml`

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write the config**

Content:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=5000]
      - id: check-merge-conflict
      - id: debug-statements
      - id: mixed-line-ending
        args: [--fix=lf]
```

- [ ] **Step 2: Install the git hook**

Run:
```bash
uv run pre-commit install
```
Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run pre-commit on all files**

Run:
```bash
uv run pre-commit run --all-files
```
Expected: passes on all files we have staged. Ruff may auto-fix a few formatting issues; if so, re-run and re-stage.

- [ ] **Step 4: Stage**

Run:
```bash
git add .pre-commit-config.yaml
```

---

## Task 17 — `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

Content:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  lint:
    name: Lint + type-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.10"
      - name: Install deps (core only, CPU-safe)
        run: |
          uv sync --extra dev --no-install-project
          uv pip install -e . --no-deps
      - name: Ruff
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check .
      - name: Pyright
        run: uv run pyright src/cleanrl_vlm || true  # warn-only at scaffold; tightened once real code lands
      - name: CPU-safe tests (imports, spec-exists)
        run: uv run pytest tests/test_imports.py tests/test_spec_exists.py -v

  tier1-smoke:
    name: Tier-1 smoke (GPU)
    runs-on: [self-hosted, gpu]
    if: false   # enable once a self-hosted GPU runner is registered for this repo
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.10"
      - name: Install deps
        run: |
          uv sync --extra dev
          uv pip install flash-attn==2.7.4.post1 --no-build-isolation
      - name: Tier-1 smoke
        run: uv run pytest -m tier1 -v
        timeout-minutes: 20

  docs:
    name: Docs build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.10"
      - name: Install deps
        run: uv sync --extra dev --no-install-project
      - name: mkdocs build --strict
        run: uv run mkdocs build --strict
```

- [ ] **Step 2: Stage**

Run:
```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
```

---

## Task 18 — `.github/workflows/docs.yml` (GitHub Pages deploy)

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Write the workflow**

Content:

```yaml
name: Docs deploy

on:
  push:
    branches: [master]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.10"
      - name: Install deps
        run: uv sync --extra dev --no-install-project
      - name: mkdocs gh-deploy
        run: uv run mkdocs gh-deploy --force
```

- [ ] **Step 2: Stage**

Run:
```bash
git add .github/workflows/docs.yml
```

---

## Task 19 — `mkdocs.yml` + `docs/index.md`

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`

- [ ] **Step 1: Write `mkdocs.yml`**

Content:

```yaml
site_name: cleanrl-vlm
site_description: Online RL finetuning of Vision-Language Models in interactive visual environments.
repo_url: https://github.com/roger-creus/cleanrl-vlm
docs_dir: docs
strict: true

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.highlight
    - content.code.copy
  palette:
    - scheme: default
      primary: black
      accent: indigo

markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.inlinehilite

nav:
  - Home: index.md
  - Architecture: ARCHITECTURE.md
  - Algorithms: ALGORITHMS.md
  - Environments: ENVS.md
  - Backbones: BACKBONES.md
  - Recipes: RECIPES.md
  - Results: RESULTS.md
  - Research: RESEARCH.md
  - Invariants: INVARIANTS.md
  - Checkpointing: CHECKPOINTING.md
  - Logging: LOGGING.md
  - Contributing: CONTRIBUTING.md
  - Troubleshooting: TROUBLESHOOTING.md
```

- [ ] **Step 2: Write `docs/index.md`**

Content:

```markdown
# cleanrl-vlm

CleanRL-style library + research paper for online RL finetuning of Vision-Language Models in interactive visual environments (ViZDoom, Atari, Minigrid).

## Quickstart

See the [README](../README.md).

## Pages

- [Architecture](ARCHITECTURE.md) — library layout, rollout/train split, LoRA dual-adapter pattern.
- [Algorithms](ALGORITHMS.md) — per-algorithm math + code pointers (populated as canon trainers land).
- [Environments](ENVS.md) — env catalogue with target scores.
- [Backbones](BACKBONES.md) — supported VLMs with memory footprint notes.
- [Recipes](RECIPES.md) — copy-pasteable reproduction commands.
- [Results](RESULTS.md) — live benchmark dashboard (auto-generated).
- [Research](RESEARCH.md) — research journal.
- [Invariants](INVARIANTS.md) — correctness invariants Inv-1..Inv-15.
- [Checkpointing](CHECKPOINTING.md) — save/resume details.
- [Logging](LOGGING.md) — rich/wandb/CSV interplay + metric glossary.
- [Contributing](CONTRIBUTING.md) — onboarding rituals for new envs/algos/backbones.
- [Troubleshooting](TROUBLESHOOTING.md) — common failure modes.

## Status

At the bootstrap milestone the library is a runnable scaffold — `pytest -m tier1 -v` passes a "hello VLM" smoke test that loads Qwen3.5-VL-0.8B and generates once. The autonomous `/loop` agent takes over from here per master-spec §13.

## Master spec

The single source of truth: [`docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`](superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md).
```

- [ ] **Step 3: Verify mkdocs builds**

Run:
```bash
uv run mkdocs build --strict
```
Expected: builds without warnings. If warnings about missing nav links appear, the Task-20 doc stubs will fix them — do that task, then re-run.

- [ ] **Step 4: Stage**

Run:
```bash
git add mkdocs.yml docs/index.md
```

---

## Task 20 — Doc stubs referenced by `mkdocs.yml` nav

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/ALGORITHMS.md`
- Create: `docs/ENVS.md`
- Create: `docs/BACKBONES.md`
- Create: `docs/RECIPES.md`
- Create: `docs/RESULTS.md`
- Create: `docs/RESEARCH.md`
- Create: `docs/INVARIANTS.md`
- Create: `docs/CHECKPOINTING.md`
- Create: `docs/LOGGING.md`
- Create: `docs/CONTRIBUTING.md`
- Create: `docs/TROUBLESHOOTING.md`

Rationale: `mkdocs build --strict` fails on missing nav links. We pre-create stubs so (a) the docs build passes from day one, (b) the /loop agent has files to update incrementally per master-spec S-1, (c) onlookers can read the planned ToC.

- [ ] **Step 1: Write each doc stub**

`docs/ARCHITECTURE.md`:
```markdown
# Architecture

Placeholder — populated by /loop per master-spec §12 rituals as the library fills in.

## Planned contents

- Library layout (mirrors master-spec §2).
- Rollout / train split: vLLM serves actor LoRA for COT generation; HF+PEFT+DeepSpeed (or FSDP2) handles the gradient step.
- LoRA dual-adapter pattern (`actor` + `critic` on the same base VLM; swap via `set_adapter`).
- Configurable LoRA target-module groups (text_attn, text_mlp, text_moe, vision_attn, vision_mlp, merger, lm_head).
- Hybrid Gated-DeltaNet + Gated-Attention + sparse MoE specifics for the Qwen3.5-VL backbones.
- Where each §8 invariant lives (unit test + `InvariantMonitor` hook).
```

`docs/ALGORITHMS.md`:
```markdown
# Algorithms

Placeholder. A per-algorithm page ships with each new `algos/*.py` file per master-spec S-8.

## Planned canon (master-spec §5)

| file | interface | algorithm | status |
|------|-----------|-----------|--------|
| `algos/ppo_cot.py`    | COT            | PPO (clipped surrogate + GAE) | not started |
| `algos/ppo_action.py` | action-scoring | PPO                           | not started |
| `algos/ppo_head.py`   | MLP head       | PPO                           | not started |
| `algos/grpo_cot.py`    | COT            | GRPO (group-relative)          | not started |
| `algos/grpo_action.py` | action-scoring | GRPO                           | not started |
| `algos/grpo_head.py`   | MLP head       | GRPO                           | not started |
| `algos/rloo_cot.py`    | COT            | RLOO (leave-one-out)           | not started |
| `algos/rloo_action.py` | action-scoring | RLOO                           | not started |
| `algos/rloo_head.py`   | MLP head       | RLOO                           | not started |

## Baselines (master-spec §7)

| file | what | status |
|------|------|--------|
| `baselines/cnn_ppo.py`         | from-scratch CNN PPO (non-VLM)              | not started |
| `baselines/zero_shot_vlm.py`   | pure prompting, no finetuning               | not started |
| `baselines/frozen_vlm_head.py` | frozen VLM + trainable MLP head (no LoRA)   | not started |
```

`docs/ENVS.md`:
```markdown
# Environments

Placeholder — populated as envs are onboarded per master-spec §11 S-7.

## Planned tiering

- **Tier-1** (CI smoke, 0.8B backbone, ≤ 10 min): `VizdoomBasic-v0`, `ALE/Pong-v5`, `MiniGrid-Empty-5x5-v0`.
- **Tier-2** (overnight / paper runs, 4B backbone): all remaining ViZDoom scenarios, full ALE suite, full Minigrid/BabyAI suite.

## Atari horizon

`max_episode_steps = 27000` is **fixed**. See master-spec §4 + §5.

## Target-score table

Lives in `configs/targets.yaml` once populated by /loop.
```

`docs/BACKBONES.md`:
```markdown
# Backbones

| name | params | context | thinking-mode default | notes |
|------|--------|---------|-----------------------|-------|
| `Qwen/Qwen3.5-VL-0.8B` | 0.8B | 262 144 | off (loop-prone) | Tier-1 smoke + debug |
| `Qwen/Qwen3.5-VL-4B`   | 4B   | 262 144 | **on**           | Tier-2 paper runs    |

Hybrid architecture: Gated-DeltaNet (linear attention) + Gated-Attention + sparse MoE. Both load via `AutoModelForCausalLM` + `AutoProcessor`. Apache-2.0.

Onboarding a new backbone = master-spec §11 S-6 ritual: registry entry + processor probe + §4 image-input probes + Inv-1..14 pass + `docs/backbone_probes/<name>.md`.
```

`docs/RECIPES.md`:
```markdown
# Recipes

Copy-pasteable commands to reproduce every Tier-2 green curve. Populated by `/loop` as curves land.

## Bootstrap smoke

```bash
uv sync --extra dev
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv run pytest -m tier1 -v
```

## First canon trainer (when it lands)

TBD — filled by /loop once `algos/ppo_cot.py` is merged.
```

`docs/RESULTS.md`:
```markdown
# Results

Live benchmark dashboard — **auto-generated** by `scripts/build_dashboard.py` from Tier-2 run metric CSVs.

At bootstrap there are no results yet. Populated by /loop as Tier-2 runs complete.

| env | algo | interface | backbone | seeds | status | curve |
|-----|------|-----------|----------|-------|--------|-------|
```

`docs/RESEARCH.md`:
```markdown
# Research journal

Append-only log of research-track activity (master-spec §6).

Each entry:

```
### YYYY-MM-DD — <method-slug> — {PROPOSED | RUNNING | PROMOTED | PARKED}

**Hypothesis.**
**Setup.**
**Results.**
**Lesson.**
**Run IDs.**
```

(No entries yet.)
```

`docs/INVARIANTS.md`:
```markdown
# Correctness invariants

Master-spec §8 canonical list. Each invariant = a pytest test in `tests/invariants/test_inv_<NN>.py` + a runtime `InvariantMonitor` hook.

Operating principle (master-spec §0): thresholds are signals, not CI cliffs. Hard-fail applies only to genuine correctness bugs.

| id | what it checks | binary / tunable | location |
|----|----------------|------------------|----------|
| Inv-1  | LoRA trainability split (lora_/head params train; base frozen)                          | binary   | `tests/invariants/test_inv_01.py` |
| Inv-2  | LoRA weights actually change over training                                               | tunable  | `tests/invariants/test_inv_02.py` |
| Inv-3  | Active adapter sanity (actor/critic set correctly per forward)                           | binary   | `tests/invariants/test_inv_03.py` |
| Inv-4  | Training ↔ inference logprob parity (vLLM vs HF forward)                                 | tunable  | `tests/invariants/test_inv_04.py` |
| Inv-5  | Gradient-norm correctness cross-check                                                     | binary   | `tests/invariants/test_inv_05.py` |
| Inv-6  | FP16 stability (GradScaler + NaN/Inf watch)                                              | tunable  | `tests/invariants/test_inv_06.py` |
| Inv-7  | Checkpoint round-trip (save → load → deterministic rollout matches)                       | binary   | `tests/invariants/test_inv_07.py` |
| Inv-8  | Image-input correctness (patch coverage, resolution floor, cross-rank token count)         | binary   | `tests/invariants/test_inv_08.py` |
| Inv-9  | Reward-pipeline integrity (scripted rewards survive to advantage)                          | binary   | `tests/invariants/test_inv_09.py` |
| Inv-10 | Episode-boundary masking (GAE resets at `done=True`)                                      | binary   | `tests/invariants/test_inv_10.py` |
| Inv-11 | Determinism under fixed seed (bitwise-identical re-runs)                                  | binary   | `tests/invariants/test_inv_11.py` |
| Inv-12 | Resume parity (continuous 200 ≈ 100 + resume + 100)                                       | tunable  | `tests/invariants/test_inv_12.py` |
| Inv-13 | Padding & image-token masking (zero gradient contribution)                                | binary   | `tests/invariants/test_inv_13.py` |
| Inv-14 | Distributed-broadcast agreement (LoRA weights identical across ranks)                      | binary   | `tests/invariants/test_inv_14.py` |
| Inv-15 | Ground-truth vision probe (VLM actually perceives the scene)                              | tunable  | `scripts/probe_vision.py` + `tests/invariants/test_inv_15.py` |

Test files populated by /loop as the corresponding infrastructure lands.
```

`docs/CHECKPOINTING.md`:
```markdown
# Checkpointing

Master-spec §10 describes the full save/resume system:

- Atomic write + rename + integrity manifest.
- Directory per checkpoint — model, optimizer, training, envs, logging, config, manifest.
- Retention: last 3 + every 10th + first + last-known-good.
- SIGTERM handler flushes inside 60 seconds.
- Resume gate runs Inv-7 + Inv-12 before continuing.

Implementation lives in `src/cleanrl_vlm/training/checkpoint.py` (populated by /loop).
```

`docs/LOGGING.md`:
```markdown
# Logging

Three parallel sinks (master-spec §9):

- **Rich console** — live colored dashboard, auto-off in headless/CI.
- **W&B** — opt-in via `--track`; mirrors every scalar + histograms + eval-episode videos.
- **CSV** — always on; `runs/<name>/metrics.csv` one row per step with every scalar.

Plus `runs/<name>/histograms.parquet` for distributions and `runs/<name>/manifest.json` for the run config + git SHA + pip freeze.

Implementation lives in `src/cleanrl_vlm/training/logging.py` (populated by /loop).
```

`docs/CONTRIBUTING.md`:
```markdown
# Contributing

Master-spec §11 defines 12 standing rules (S-1..S-12) that govern every PR. Quick reference:

- **S-1** — Correctness coverage never decreases. New file under `algos/`, `experimental/`, or `envs/` ships with matching tests in the same PR.
- **S-3** — Perf optimizations prove Inv-4 parity before merge.
- **S-5** — External-library changes (vllm, accelerate, deepspeed, peft, transformers, flash_attn) require a web-fetch of current docs with summary in the PR body.
- **S-6, S-7, S-8** — Backbone / env / algorithm onboarding rituals.
- **S-12** — Every PR updates `CHANGELOG.md` with { what, why, evidence, invariants-run }.

See the master spec for the full list.
```

`docs/TROUBLESHOOTING.md`:
```markdown
# Troubleshooting

Collected failure modes and diagnostic paths. Populated by /loop as issues surface.

## Install

- `flash-attn` build fails with CUDA mismatch → ensure `nvcc --version` matches the CUDA PyTorch was built against. Use `TORCH_CUDA_ARCH_LIST` env var to limit arches. Falls back to SDPA at runtime, but a build failure breaks full install.

## Runtime (populated as /loop encounters issues)

- OOM on backbone load → reduce `min_pixels`/`max_pixels` in the processor config; enable gradient checkpointing.
- Logprob drift between vLLM and HF forward (Inv-4) → check tokenizer mismatch, attention mask divergence, stale adapter on vLLM server, image preprocessing path difference.
- NaN in gradients (Inv-6) → check loss scale history; reduce LR; dump the offending microbatch from `runs/<name>/nan_dumps/`.
```

- [ ] **Step 2: Run mkdocs build --strict**

Run:
```bash
uv run mkdocs build --strict
```
Expected: builds cleanly, no warnings, produces `site/` output.

- [ ] **Step 3: Stage**

Run:
```bash
git add docs/
```

---

## Task 21 — Run full lint + test suite before commit

- [ ] **Step 1: Ruff check**

Run:
```bash
uv run ruff check .
```
Expected: no errors. If any, fix (most auto-fix with `uv run ruff check --fix .`) and re-stage.

- [ ] **Step 2: Ruff format check**

Run:
```bash
uv run ruff format --check .
```
Expected: "X files already formatted." If any would be reformatted, run `uv run ruff format .`, re-stage.

- [ ] **Step 3: Pyright (warn-only for scaffold)**

Run:
```bash
uv run pyright src/cleanrl_vlm
```
Expected: 0 errors (scaffold is mostly docstrings + empty modules). Warnings tolerable.

- [ ] **Step 4: Full test suite (CPU-safe subset)**

Run:
```bash
uv run pytest tests/test_imports.py tests/test_spec_exists.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Tier-1 smoke if GPU available**

Run:
```bash
uv run pytest -m tier1 -v
```
Expected on GPU machine: PASS (takes 1-2 min for hello-VLM). On CPU-only: SKIPPED.

- [ ] **Step 6: Pre-commit dry run**

Run:
```bash
uv run pre-commit run --all-files
```
Expected: all hooks pass. If any auto-fix fires, re-stage the changes.

- [ ] **Step 7: Verify git status is clean of unintended changes**

Run:
```bash
git status
```
Expected: `.git/` + staged changes only; no unstaged modifications; no untracked files (beyond those intentionally ignored by `.gitignore`).

---

## Task 22 — Initial bootstrap commit

- [ ] **Step 1: Review the staged diff once**

Run:
```bash
git diff --staged --stat
```
Expected: many new files across `src/`, `tests/`, `docs/`, `.github/`, plus top-level `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`, `LICENSE`, `LOOP_STATE.md`, `AUTONOMY_LOG.md`, `CHANGELOG.md`, `MEMORY.md`, `.gitignore`, `.pre-commit-config.yaml`, `mkdocs.yml`. Sanity-check there is nothing unintended (e.g., `.venv/`, large binary files).

- [ ] **Step 2: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
Bootstrap scaffold — v0.0.1

Moves prior prototype to branch `old` and starts a fresh master with:

* pyproject.toml + UV (Python 3.10–3.12, torch 2.6.0, transformers from git,
  peft, accelerate, deepspeed, vllm, gymnasium + vizdoom + minigrid + ale)
* Directory skeleton per master-spec §2 (src/cleanrl_vlm/*, algos/,
  experimental/, baselines/, configs/, prompts/, scripts/, tests/, docs/)
* CLAUDE.md carries master-spec §13 autonomy contract verbatim at top
* LICENSE (MIT), README, CHANGELOG, LOOP_STATE, AUTONOMY_LOG, MEMORY stubs
* Docs scaffold: mkdocs.yml + docs/index.md + 12 nav stubs
* CI: ruff + pyright + pytest (CPU) + tier1 smoke (self-hosted GPU, gated off
  until runner registered) + mkdocs build; docs deploy workflow
* Pre-commit config (ruff + standard hooks)
* Tests: imports sanity + spec-exists tripwire + hello-VLM GPU smoke that
  loads Qwen/Qwen3.5-VL-0.8B and generates once (proves dep graph + model
  loading works end-to-end)

Master spec: docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md
Plan: docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md
Next: LOOP_STATE.md task B-ppo-cot-vizdoom-basic-0.8B (autonomous /loop
takes over per master-spec §13).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit created. Pre-commit hooks run during the commit and must all pass. If a hook autofixes something, the commit aborts — re-stage the fix and re-run the commit.

- [ ] **Step 3: Verify**

Run:
```bash
git log --oneline -3
git log old --oneline -3
git branch -a
```
Expected:
- Current branch (`scaffold-bootstrap` from the orphan step, or already moved back to `master`) has the bootstrap commit at top.
- `old` has the preserved prototype tip.
- Both branches visible in `git branch -a`.

- [ ] **Step 4: Fast-forward `master` to the scaffold commit**

Run:
```bash
git branch -f master scaffold-bootstrap
git checkout master
git branch -d scaffold-bootstrap
```
Expected: `master` now points at the new scaffold commit; `scaffold-bootstrap` branch is deleted; working tree unchanged.

- [ ] **Step 5: Final verification**

Run:
```bash
git branch --show-current
git log --oneline -1
git status
```
Expected: on `master`; HEAD is the scaffold commit; clean working tree.

---

## Task 23 — Smoke-run the scaffold end-to-end

- [ ] **Step 1: Re-sync from a clean env (to verify the locked deps work from scratch)**

Run:
```bash
uv sync --extra dev
```
Expected: "Audited N packages" — no net install changes if `uv.lock` is honoured.

- [ ] **Step 2: Run the hello-VLM smoke test (if GPU available)**

Run:
```bash
uv run pytest tests/smoke/test_hello_vlm.py -v
```
Expected on GPU: PASS. If the model hasn't been downloaded yet, it downloads on first run (~1.5GB).

- [ ] **Step 3: Full test suite**

Run:
```bash
uv run pytest -v
```
Expected: all tests PASS (or SKIP with clear reasons where hardware is unavailable).

- [ ] **Step 4: Mkdocs build**

Run:
```bash
uv run mkdocs build --strict
```
Expected: clean build to `site/`.

- [ ] **Step 5: Lint**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
```
Expected: no errors, no unformatted files.

---

## Task 24 — Hand off to the autonomous /loop

- [ ] **Step 1: Verify `LOOP_STATE.md` points at the next task**

Read `LOOP_STATE.md`; confirm the "Next task" section targets `B-ppo-cot-vizdoom-basic-0.8B` and the prioritized queue lists B through M.

- [ ] **Step 2: Verify `CLAUDE.md` carries §13 at the top**

Read first 100 lines of `CLAUDE.md`; confirm `## AUTONOMOUS OPERATION CONTRACT (master-spec §13 verbatim)` is present and the "Never stop" / "Never ask the user anything" rules are visible above the repo orientation.

- [ ] **Step 3: Print the kickoff instructions for the human**

The bootstrap is complete. The human starts the autonomous `/loop` session with:

```
/loop

You are the autonomous /loop agent for cleanrl-vlm. Master spec at
docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md — read the
full file at the start of every iteration. Follow §13 strictly. Use the
superpowers skills continuously (brainstorming, writing-plans,
subagent-driven-development, code-reviewer, simplify, systematic-debugging,
verification-before-completion, test-driven-development). Never ask me
questions. Never stop except on Ctrl-C. All decisions go in AUTONOMY_LOG.md.
```

From here, the /loop iterates forever per master-spec §13.3 until the human Ctrl-C's. The first iteration reads `LOOP_STATE.md`, sees task `B-ppo-cot-vizdoom-basic-0.8B`, brainstorms it, uses writing-plans to produce `docs/superpowers/specs/plans/YYYY-MM-DD-ppo-cot-vizdoom-basic.md`, executes via subagent-driven-development, verifies, reviews, simplifies, commits. Then picks the next task.

---

## Plan self-review (mandatory per writing-plans skill)

1. **Spec coverage.** The spec is §0–§14. This plan implements master-spec §2 (repo layout scaffold), §1 "Git strategy" (old-branch preservation), §13.6/§13.7 (`CLAUDE.md` + kickoff prompt), §12 (docs stubs), minimal §9 (CI scaffolding). All other sections (§3 model stack, §4 envs, §5 algorithms, §6 research, §7 baselines, §8 invariants, §10 checkpoint, §11 standing rules enforcement) are **intentionally out of scope for this plan** — they are the /loop's responsibility per §13.2, tracked in `LOOP_STATE.md`'s prioritized queue. No gaps to fix.

2. **Placeholder scan.** No `TBD` / `TODO` / "implement later" / "fill in details" / "add appropriate error handling" / "similar to Task N" / steps without code or commands. The doc stubs in Task 20 carry the phrase "Placeholder — populated by /loop" — that is explicit and correct (the stubs exist to satisfy `mkdocs --strict` and to give the /loop agent targets to update incrementally, not as deferred work in this plan).

3. **Type consistency.** No cross-task function signatures to check — this plan is entirely scaffolding (config files, docs, import-sanity tests, hello-VLM smoke). No cleanrl-vlm types defined yet.

No issues found.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review between tasks.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

**Which approach?**
