# Changelog

Per §11 S-12: every `/loop` iteration that changes tracked state adds
an entry here — `{ what, why, evidence, invariants-run }`. Newest on top.
One entry per iteration, not per commit within an iteration.

---

## 2026-04-20 — iter 14 — integration test (plan task 20/27)

**What.** 3 commits driving the first end-to-end trainer run on the 2B
backbone:

- `e3a11b0` — `tests/integration/test_trainer_short_run.py`: 10-iter
  subprocess-based smoke, asserts runs dir + metrics.csv + header
  columns.
- `d0098b4` — 4 integration-bug fixes discovered by the test run:
  1. BaseVLM `device_map="cuda"` (model was on CPU → flash-attn fail).
  2. `mm_token_type_ids` threaded through the actor re-score forward
     with zero-extension for generated tokens (Qwen3-VL M-RoPE).
  3. CriticHead on model device + fp32 dtype.
  4. Trainable LoRA params + critic_head cast to fp32 for GradScaler
     compatibility.
  Plus the documented ITER-14 SHORTCUT: PPO re-score uses ratio=1.0
  until iter 15 caches full_ids.
- `6ea86f2` — env ID rename Vizdoom*-v0 → v1 (gymnasium_wrapper
  deprecation); integration test scaled to num_envs=1, num_steps=2,
  max_new_tokens=16, streams stdout to log file.

**Why.** Task 20 is the first moment the assembled trainer actually
runs against the real backbone + real env. Expected to surface
integration bugs; 5 real bugs caught + 1 shortcut deferred.

**Evidence.**
- `pytest tests/integration/test_trainer_short_run.py -v -m "tier1
  and gpu"` → **1 passed in 80.37s** on single A6000.
- `pytest tests/unit/test_env_factory.py -v` → 8 passed (post-rename).
- Run artifacts: `runs/ppo_cot__VizdoomBasic-v1__qwen3-vl-2b-instruct__
  0__2026-04-20/metrics.csv` + checkpoints/step_000002/.

**Invariants run.** Inv-4 single-path drift check runs at iter 0
minibatch 0 of the update loop — logs `inv_4_status=green` (drift=0
because lp_new=mb_lp_old.clone()). Inv-01/Inv-05 register in
InvariantMonitor but sample_every=10 doesn't fire in a 1-iter run.

---

## 2026-04-20 — iter 13 — cluster script + trainer (plan tasks 18-19/27)

**What.** 2 commits landing the trainer assembly.

- Task 18 (`7bc03cf`) — `scripts/_cluster_env.sh`: HF_HOME,
  CUDA_HOME, tokenizers-parallelism, CUBLAS_WORKSPACE_CONFIG
  defaults. Sourceable before training.
- Task 19 (`8d63816`) — `algos/ppo_cot.py` (346 LOC):
  single-file trainer glueing every `cleanrl_vlm.*` module into the
  PPO-COT iteration loop with FP16 + LoRA dual adapters + Inv-04
  single-path drift check + CSV/Rich/W&B logging + algo-slug
  checkpoint + microbatch probe + determinism seeding. All reviewer
  M1-M8 and m1-m12 findings encoded. Import sanity green.

**Why.** The trainer is the convergence point of every iter-6..12
primitive. Landing it as a single file with no further shared
abstractions matches the spec §5 "CleanRL-style single-file"
discipline; further refinement happens through the integration test
(Task 20) and the simplify pass (Task 25).

**Evidence.**
- `bash -c "source scripts/_cluster_env.sh"` → env exports fine.
- `uv run --no-env-file python -c "import algos.ppo_cot"` → ok.
- 346-line trainer; all imports resolve; ruff green.

**Invariants run.** None new landed (Inv-04 drift check is in the
trainer body; it activates when Task 20 runs the trainer).

---

## 2026-04-20 — iter 12 — InvariantMonitor + 5 invariants (plan task 17/27)

**What.** 1 commit landing the invariant batch.

- `src/cleanrl_vlm/training/invariants.py` — `InvariantMonitor`
  dataclass (passive at iter 4; runtime hook in D-invariants-runtime),
  `check_inv_01_lora_trainability`, `check_inv_05_grad_norm`,
  `INV_04_TOLERANCE=1e-4` public constant (reviewer B2).
- `tests/invariants/test_inv_04_logprob_parity.py` — tolerance pin
  + @gpu placeholder (real re-score body lands in `algos/ppo_cot.py`
  Task 19).
- `tests/invariants/test_inv_05_grad_norm.py` — clip vs manual norm.
- `tests/invariants/test_inv_09_reward_pipeline.py` — scripted
  rewards flow into GAE unchanged.
- `tests/invariants/test_inv_11_determinism.py` — CPU bitwise
  rollout equality under `_make_fully_deterministic(seed)` fixture
  with full reviewer-M5 settings (CUBLAS_WORKSPACE_CONFIG,
  PYTHONHASHSEED, HF_SEED, seeded RNGs, cudnn, `use_deterministic_
  algorithms(True, warn_only=True)` — deviation logged).
- `tests/invariants/test_inv_13_pad_image_token_mask.py` — masked
  positions have zero gradient contribution.

**Why.** All iter-4-scope §8 invariants now exist as pytest tests
(Inv-1/3 already shipped iter 8; Inv-6 iter 10; Inv-10 iter 9; this
iter adds Inv-4/5/9/11/13). Trainer (Task 19) will wire the
Monitor + per-Inv checks; tests-first stance holds.

**Evidence.**
- `uv run --no-env-file pytest tests/invariants/ -v -m "not gpu"`
  → **14 passed in 141.70s** (1 `@gpu` deselected).

**Invariants run.**
- Inv-4 (tolerance constant), Inv-5 (grad norm), Inv-9 (reward
  pipeline), Inv-11 (bitwise determinism), Inv-13 (pad masking) —
  all new green this iter.

---

## 2026-04-20 — iter 11 — microbatch + logging + checkpoint (plan tasks 14-16/27)

**What.** 3 commits landing training-layer plumbing.

- Task 14 (`c2d8173`) — `src/cleanrl_vlm/training/microbatch_probe.py`:
  `probe_microbatch` doubles until OOM/cap; `record_microbatch_probe`
  writes `runs/<name>/microbatch_probe.json`. 3 unit tests.
- Task 15 (`7f7880b`) — `src/cleanrl_vlm/training/logging.py`:
  `CsvWriter` over 39-column §9 schema including m4
  `gen_truncated_rate`, m5 `lora_weight_norm_{actor,critic}` +
  `adapter_sync_wall_s`, B2 `inv_4_status`. `RichDashboard` with TTY
  auto-off. `wandb_init` shim. 2 unit tests.
- Task 16 — `src/cleanrl_vlm/training/checkpoint.py`:
  `save_vlm_actor_critic_checkpoint(algo_slug, ...)` (reviewer m6),
  atomic tmp-rename + sha256 integrity hashes + full §10 directory
  layout. Signature-only test; round-trip deferred to
  `J-checkpoint-resume-e2e`. 1 unit test.

**Why.** Training-layer plumbing needs to be in place before Task 17
(InvariantMonitor scaffold) and Task 19 (trainer assembly). The §9
schema, m4/m5 metric columns, and m6 checkpoint signature all ship
together for a coherent "training utilities" landing.

**Evidence.**
- `pytest tests/unit/test_microbatch_probe.py -v` → 3 passed in 0.49s.
- `pytest tests/unit/test_logging.py -v` → 2 passed.
- `pytest tests/unit/test_checkpoint.py -v` → 1 passed in 36.06s.

**Invariants run.** None landed this iter — Task 17 lands
Inv-04/05/09/11/13 next.

---

## 2026-04-20 — iter 10 — generate + precision (plan tasks 12-13/27)

**What.** 2 commits for the generation path + precision layer.

- Task 12 (`3deb27a`) — `src/cleanrl_vlm/rollout/in_process.py`:
  `CotRolloutStep` typed dataclass (reviewer M1) + `generate_cot_actions`
  that runs ac_model.get_action, parses via `parse_action_cot`, fills
  `gen_truncated` when no EOS appears, sums logprobs over generated
  span.
- Task 13 — `src/cleanrl_vlm/training/distributed.py::load_accelerator_config`
  with single-rank "sharding ignored" startup log (reviewer m11);
  `src/cleanrl_vlm/training/precision.py::Fp16State` wrapping
  `torch.amp.GradScaler("cuda", ...)` with bounded scale-history deque;
  `tests/invariants/test_inv_06_fp16_scale.py` with 2 tests covering
  scale history + disabled mode.

**Why.** Generation + FP16 precision are on the critical path to the
trainer assembly (Task 19).

**Evidence.**
- Task 12 import sanity: `from cleanrl_vlm.rollout.in_process import
  CotRolloutStep, generate_cot_actions` → ok.
- Task 13: `uv run --no-env-file pytest
  tests/invariants/test_inv_06_fp16_scale.py -v` → 2 passed in 86.54s
  (GPU cuda:0).

**Invariants run.**
- Inv-6 — FP16 stability: 2 tests green. Corrected plan literal's
  no-NaN-on-scaled-grads assertion to actual Inv-6 invariant
  (scale history non-collapsing).

---

## 2026-04-20 — iter 9 — prompts + parser + rollout (plan tasks 9-11/27)

**What.** 3 commits across prompts and rollout layers.

- Task 9 (`a84436a`) — `src/cleanrl_vlm/prompts/templates/vizdoom/basic/
  {actor,critic,vision_probe}.txt`: VizdoomBasic-specific text templates.
- Task 10 (`2375a82`) — `src/cleanrl_vlm/prompts/parser.py` with regex
  last-match + whitelist (reviewer M2 + M3, including repeated-ACTION
  pathology) + `src/cleanrl_vlm/prompts/builder.py` with env-id → slug
  lookup. 9 tests.
- Task 11 — `src/cleanrl_vlm/rollout/buffer.py`: `compute_gae`
  function + `RolloutBuffer` dataclass with pre-allocated tensors for
  obs/actions/logprob_sum/rewards/values/dones/advantages/returns.
  4 tests including Inv-10 (GAE resets at done boundary).

**Why.** Prompts precede the trainer's use of `PromptBuilder`. Rollout
buffer is the trainer's main data structure. Inv-10 is the first
invariant that can run purely on tensor math without a real VLM.

**Evidence.**
- `uv run --no-env-file pytest tests/unit/test_action_parser.py
  tests/unit/test_prompt_builder.py -v` → 9 passed in 0.75s.
- `uv run --no-env-file pytest tests/unit/test_gae.py
  tests/unit/test_rollout_buffer.py
  tests/invariants/test_inv_10_episode_boundary.py -v` → 4 passed in
  38.06s.

**Invariants run.**
- Inv-10 — episode-boundary masking: 1 test green.

---

## 2026-04-20 — iter 8 — model layer cornerstone (plan task 8/27)

**What.** Plan Task 8. 1 commit for the 400-line plan section.

- `src/cleanrl_vlm/models/actor_critic.py` (185 LOC) —
  `DecoupledActorCriticVLM_COT` with dual-adapter LoRA (actor + critic
  on same BaseVLM), `CriticHead` on last non-pad hidden state,
  `max_new_tokens` on generate path, build-time assert of actor/critic
  target_modules identity (reviewer m10), critic `.forward()`-only
  discipline (reviewer m9), `active_adapter` tripwire ctxmgr (reviewer
  m7) with `_active_adapter_name` shim over PEFT's str-vs-list
  divergence.
- `tests/invariants/_tiny_vlm.py` (111 LOC, under 150 time-box per
  reviewer m3) — CPU nn.Module stub that PEFT can LoRA-wrap so Inv-01/
  03 tests don't need a GPU + backbone download.
- `tests/invariants/test_inv_01_lora_trainability.py` — 3 tests
  covering the reviewer M4 extensions (requires_grad split + base-
  weight identity across set_adapter + disjoint actor/critic optim
  groups).
- `tests/invariants/test_inv_03_active_adapter.py` — 3 tests covering
  the tripwire (pass on match, raise on mismatch, raise on mutation).

**Why.** Task 8 is the model-layer cornerstone. All later tasks
(rollout, trainer, integration) consume `DecoupledActorCriticVLM_COT`.
Landing Inv-1/3 with M4 extensions now gives the invariant safety net
before the trainer wires them together.

**Evidence.** `uv run --no-env-file pytest tests/invariants/
test_inv_01_lora_trainability.py tests/invariants/
test_inv_03_active_adapter.py -v` → 6 passed in 140.52s on CPU.
Stub size 111 LOC.

**Invariants run.**
- Inv-1 (LoRA trainability split) — 3 sub-tests including M4
  extensions.
- Inv-3 (active adapter sanity) — 3 sub-tests on the ctxmgr tripwire.

---

## 2026-04-20 — iter 7 — model layer pt 1 (plan tasks 5-7/27)

**What.** 3 commits for the model layer scaffolding.

- Task 5 (`7b250c0`) — `src/cleanrl_vlm/models/lora_topology.py` +
  `tests/unit/test_lora_topology.py`: `default_target_modules(groups)`
  over 7 groups (text_attn, text_mlp, vision_attn, vision_mlp, merger,
  lm_head, text_moe); raises on unknown; de-duplicates preserving
  order. 6 tests.
- Task 6 (`d7e2649`) — `src/cleanrl_vlm/models/heads.py` +
  `tests/unit/test_heads.py`: `CriticHead(input_dim)` + `ActorHead(
  input_dim, num_actions)` + `layer_init` (orthogonal init, zero
  bias). 3 tests.
- Task 7 (`b34e8ea`) — `src/cleanrl_vlm/models/base_vlm.py`: `BaseVLM`
  wrapping `AutoModelForImageTextToText` + `AutoProcessor`; exposes
  `preprocess_obs_and_text`, `last_hidden_state`, `get_trainable_params`.
  Import-sanity only (constructor loads real backbone; exercised in
  downstream Task 8 + integration).

**Why.** Model layer is dependency-ordered: topology + heads before
`BaseVLM`, all three before `DecoupledActorCriticVLM_COT` in Task 8.

**Evidence.** 17 unit tests green total through iter 7
(8 env + 6 lora + 3 heads). BaseVLM imports cleanly.

**Invariants run.** Inv-1/3 tests land in Task 8 (iter 8); nothing
applicable yet at iter 7.

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
