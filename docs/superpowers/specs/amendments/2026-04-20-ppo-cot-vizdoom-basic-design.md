---
title: PPO-COT on VizdoomBasic-v0 with Qwen3-VL-2B-Instruct — Design Spec
date: 2026-04-20
authors: Claude (/loop agent, iter 4)
slug: B-ppo-cot-vizdoom-basic-2B
status: self-approved per §13.1 (no human approval gate in autonomous mode)
reviewed-by: TBD — code-reviewer subagent (run before first code commit)
refs:
  - docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md (master spec §2 §3 §5 §8 §9)
  - docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md (bootstrap plan)
  - docs/superpowers/specs/amendments/2026-04-20-backbone-names-correction.md (backbone naming)
---

# PPO-COT · VizdoomBasic · Qwen3-VL-2B-Instruct — Design

This design spec is the source of truth for task **`B-ppo-cot-vizdoom-basic-2B`** — the first canon trainer. A detailed per-task implementation plan follows in `docs/superpowers/specs/plans/2026-04-20-ppo-cot-vizdoom-basic.md` (produced by the writing-plans skill in a subsequent /loop iteration).

---

## §1. Goal

Produce `algos/ppo_cot.py`: a single-file PPO trainer that finetunes
`Qwen/Qwen3-VL-2B-Instruct` on `VizdoomBasic-v0` via the **chain-of-thought
(COT)** action interface. The VLM generates free-form text; the trainer
parses `ACTION: <NAME>` from the tail; the token-level sum-logprob over
the generated span feeds the PPO loss. A separate critic LoRA adapter on
the same base weights + an MLP `CriticHead` provides `V(s)` for GAE.

**Green = agent judges (§0 / §9):**

- Mean episodic return over the last 10 % of the training window is
  materially above the zero-shot-VLM baseline's return. No target score is
  asserted as a hard threshold — agent reads the curve.
- §8 invariants Inv-1, Inv-3, Inv-5, Inv-6, Inv-9, Inv-10, Inv-11, Inv-13
  pass throughout the run (Inv-2/4/7/8/12/14/15 are landed as separate
  tasks; see §6 below).
- No NaN / Inf events. Grad-norm band stays consistent. Loss-scale
  history does not collapse.
- Reproducibility: `--seed 0` re-run matches the first run's curve to
  within agent-judged tolerance (effectively binary, since Inv-11
  asserts bitwise equality on the rollout tensor).

## §2. Scope

**In scope** (iter 4+, multiple /loop cycles):

1. Env layer port → `src/cleanrl_vlm/envs/`.
2. Model layer port → `src/cleanrl_vlm/models/`.
3. Prompt layer port → `src/cleanrl_vlm/prompts/` + `prompts/vizdoom/basic/`.
4. Rollout buffer + GAE → `src/cleanrl_vlm/rollout/`.
5. Training helpers (distributed, precision, logging, invariants)
   → `src/cleanrl_vlm/training/`.
6. `algos/ppo_cot.py` single-file trainer.
7. Per-env config: `configs/envs/VizdoomBasic-v0.yaml`, plus
   `configs/backbones.yaml` + `configs/targets.yaml`.
8. Tests: unit + invariant (the eight listed above) + tier1 integration.
9. Ground-truth vision probe → `scripts/probe_vision.py` +
   `prompts/vizdoom/basic/vision_probe.txt` + auto-generated
   `docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md`.
10. Backbone probe artifact → `docs/backbone_probes/qwen3-vl-2b-instruct.md`.
11. Docs: `docs/ALGORITHMS.md` PPO-COT paragraph + `docs/ENVS.md` row +
    `docs/RECIPES.md` entry + `docs/BACKBONES.md` probe link.

**Out of scope** (tracked as follow-up tasks in `LOOP_STATE.md`):

- **vLLM rollout path** → `E-vllm-rollout-path`. iter 4 uses in-process HF
  generation, which is slow but simple and side-steps the adapter-sync
  complexity of vLLM. §3 §14 of the master spec are satisfied in a later
  cycle once the in-process trainer's correctness is established.
- **InvariantMonitor runtime hook** → `D-invariants-runtime`. iter 4 runs
  invariants only as pytest tests, not continuously inside the training
  loop.
- **Full checkpoint/resume** → `J-checkpoint-resume-e2e`. iter 4 checkpoints
  LoRA adapters + critic head + optimizer state + RNG snapshot in a
  simple format that `scripts/resume.py` can load; Inv-7 / Inv-12 are
  left as sanity-log fields until `J`.
- **Distributed multi-rank** → iter 4 runs single-rank (`num_processes=1`)
  for simplicity. Inv-14 lands in `D` when we scale to 2+ ranks.
- **Inv-8 image-input correctness probe** → runs as a one-shot at backbone
  onboarding (this iter) but the runtime-invariant monitor lives in `D`.
- **Reference canon trainers** `ppo_action.py`, `ppo_head.py`, and the six
  GRPO / RLOO variants → `F-canon-expand`. iter 4 ships only `ppo_cot.py`
  to validate the design end-to-end; the action-scoring and MLP-head
  variants will re-use `src/cleanrl_vlm/` modules and land incrementally.
- **Frame-stack layout** → iter 4 uses a *single frame* per obs (not the
  4-frame horizontal tile from master-spec §4) to minimize variables on
  the first trainer. Frame-stacking becomes a first-class ablation in
  iter-5+ once the single-frame baseline is green.
- **LoRA topology auto-probe** → iter 4 hard-codes the spec's default
  group set (text_attn + text_mlp + lm_head). The group-enable ablation
  (`runs/<name>/lora_topology_probe.json`) happens in `F-canon-expand` or
  a dedicated ablation cycle.
- **FSDP2 sharding** → iter 4 is single-rank FP16 on one A6000; DS-ZeRO-2
  and FSDP2 are moot. `--sharding` flag accepted but ignored at
  `num_processes=1`.
- **Prompt templates for other ViZDoom scenarios** → `C-envs-tier1-expand` +
  `I-envs-tier2-full`.

## §3. Architecture (hybrid library + single-file trainer)

The master spec's §2 / §5 prescribe CleanRL-like single-file trainers
backed by a shared library. This section locks the **concrete module
split** before code lands, so subsequent ports and the six future canon
trainers can slot in without re-architecting.

```text
algos/ppo_cot.py                        single-file trainer (~600–800 lines)
├── Args (tyro dataclass)               hyperparams + CLI flags
├── main()                              iteration loop
├── rollout phase                       in-process HF generation
├── advantage compute                   GAE
├── PPO update                          clipped surrogate + value + entropy
├── logging                             rich + wandb + CSV
├── checkpoint save                     LoRA + critic head + optim + RNG
└── uses → src/cleanrl_vlm/...

src/cleanrl_vlm/
├── envs/
│   ├── registry.py                     make_env(env_id, seed, idx, run_name)
│   ├── vizdoom/
│   │   ├── __init__.py
│   │   ├── factories.py                make_vizdoom_env(env_id, config)
│   │   └── action_tables.py            per-scenario button name table
│   └── wrappers.py                     FrameSkipEnv, ScreenOnlyWrapper, DiscreteMultiBinaryWrapper
├── models/
│   ├── base_vlm.py                     BaseVLM wrapping AutoModelForImageTextToText
│   ├── lora_topology.py                default_target_modules(), LoRA group enable/disable
│   ├── heads.py                        CriticHead, ActorHead (+ layer_init)
│   └── actor_critic.py                 DecoupledActorCriticVLM_COT (dual-adapter orchestration)
├── prompts/
│   ├── builder.py                      PromptBuilder: assemble messages, apply chat template
│   └── templates/
│       └── vizdoom/
│           ├── basic/
│           │   ├── actor.txt           COT actor system prompt
│           │   ├── critic.txt          critic system prompt
│           │   └── vision_probe.txt    Inv-15 probe questions
│           └── (corridor, defend_line, ... for C + I)
├── rollout/
│   ├── buffer.py                       RolloutBuffer (tensors) + GAE
│   └── in_process.py                   generate_cot_actions(model, processor, obs_batch, prompts)
└── training/
    ├── distributed.py                  accelerate/DS config loader + active-adapter context mgr
    ├── precision.py                    FP16 GradScaler wrapper + Inv-6 loss-scale watcher
    ├── checkpoint.py                   save(dir) / load(dir) for actor+critic LoRA + heads + optim + RNG
    ├── logging.py                      RichDashboard + CSV writer + W&B shim + manifest.json
    ├── invariants.py                   InvariantMonitor scaffold (passive until D) + per-invariant checks
    └── args_base.py                    shared CLI fields re-used across canon trainers

tests/
├── unit/
│   ├── test_action_parser.py           parse ACTION: <NAME> from COT tail
│   ├── test_env_factory.py             make_env returns Gymnasium env of correct shape
│   ├── test_prompt_builder.py          chat template output shape + image-token presence
│   ├── test_gae.py                     GAE on synthetic rollout vs hand-computed
│   ├── test_rollout_buffer.py          buffer rolling indices + tensor dtypes
│   └── test_lora_topology.py           target module resolution against a mini-MLP fixture
├── invariants/
│   ├── test_inv_01_lora_trainability.py
│   ├── test_inv_03_active_adapter.py
│   ├── test_inv_05_grad_norm.py
│   ├── test_inv_06_fp16_scale.py
│   ├── test_inv_09_reward_pipeline.py
│   ├── test_inv_10_episode_boundary.py
│   ├── test_inv_11_determinism.py
│   └── test_inv_13_pad_image_token_mask.py
├── integration/
│   └── test_trainer_short_run.py       tier1 @gpu — 10 iters on VizdoomBasic
└── smoke/
    └── test_hello_vlm.py               already in iter 2/3; unchanged

scripts/
├── probe_vision.py                     Inv-15 ground-truth vision probe
└── _cluster_env.sh                     CUDA_HOME + HF_HOME boilerplate

configs/
├── backbones.yaml                      per-backbone defaults (pixel budget, attn_impl, thinking)
├── targets.yaml                        reference target scores per env
└── envs/
    └── VizdoomBasic-v0.yaml            frame_skip, resolution, reward scaling, max_episode_steps
```

### Module boundaries — what each unit does, inputs, outputs, deps

Every unit below has **one clear purpose**, a **typed interface**, and
**deps only on its own layer + layers below**. This is the boundary
discipline called out by the brainstorming skill.

| Unit | Does | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| `envs/registry.py::make_env(env_id, config, seed, idx, run_name) → Callable[[], gym.Env]` | Single factory dispatching to `vizdoom/` / `atari/` / `minigrid/` subpackages by env-id prefix; returns a thunk for AsyncVectorEnv. | env id, per-env YAML config, seed, idx, run_name | thunk | `vizdoom.factories`, `wrappers`, gymnasium, vizdoom |
| `envs/vizdoom/factories.py::make_vizdoom_env(env_id, config) → Callable` | Apply vizdoom-gymnasium wrapper, frame-skip, action wrapper, screen-only wrapper, `RecordEpisodeStatistics`. | env id + YAML cfg | thunk | `action_tables`, `wrappers`, vizdoom.gymnasium_wrapper |
| `envs/vizdoom/action_tables.py::action_tables: dict[str, list[str]]` | Per-scenario button-name lookup. Starts with `VizdoomBasic-v0 → ["MOVE_LEFT","MOVE_RIGHT","ATTACK"]`; corridor + defend_line carried over from prototype. | — | `{env_id: [button_name, ...]}` | — |
| `envs/wrappers.py::FrameSkipEnv`, `::ScreenOnlyWrapper`, `::DiscreteMultiBinaryWrapper` | Gymnasium wrappers. `DiscreteMultiBinaryWrapper(button_names: list[str])` generalizes `DeadlyCorridorActionWrapper` / `DefendTheLineActionWrapper`. | gym.Env | gym.Env | gymnasium |
| `models/base_vlm.py::BaseVLM(vlm_name, min_pixels, max_pixels, attn_impl) → nn.Module` | Wraps `AutoModelForImageTextToText.from_pretrained` + `AutoProcessor`. Holds `preprocess_obs_and_text`, `last_hidden_state(hidden, attention_mask)`. | backbone id, pixel budget, attn impl | module with `self.model`, `self.processor` | transformers, torch |
| `models/lora_topology.py::default_target_modules(groups: set[str]) → list[str]` | Map group names (text_attn / text_mlp / vision_attn / vision_mlp / merger / lm_head / text_moe) to PEFT target-module patterns. | group set | list of module suffixes | — |
| `models/heads.py::CriticHead(input_dim)`, `::ActorHead(input_dim, num_actions)`, `::layer_init` | Orthogonal-init MLPs: 3-layer 512-hidden LeakyReLU. | hidden dim | nn.Module forward(hidden) → value | torch |
| `models/actor_critic.py::DecoupledActorCriticVLM_COT(vlm_name, lora_r, lora_alpha, lora_dropout, groups) → nn.Module` | Dual-adapter orchestration. Exposes `get_action(obs, prompts)` (generate → (full_ids, prompt_lens, texts, per-token-logprobs)) and `get_value(obs, prompts)` (critic-adapter forward → critic-head V). Enforces active-adapter invariant (Inv-3). | backbone, LoRA params | actor+critic LoRA on same base weights + CriticHead | BaseVLM, heads, lora_topology, peft |
| `prompts/builder.py::PromptBuilder(env_id, action_names)` | `.actor_prompt_for(obs_text_state) → str`; `.critic_prompt_for(obs_text_state) → str`; `.parse_action(cot_text) → int`. Reads templates from `templates/vizdoom/<slug>/*.txt`. | env id, action names | textual messages ready to chat-template | — |
| `rollout/buffer.py::RolloutBuffer(num_envs, num_steps, obs_space, action_space)` | Pre-allocated tensors: obs, action, logprob_sum, reward, value, done, advantage, return. `.compute_gae(gamma, lam, next_value, next_done)`. | shapes | tensor dict by step | torch |
| `rollout/in_process.py::generate_cot_actions(ac_model, obs_batch, prompt_texts, max_new_tokens) → (actions, full_ids, per_token_logprobs, prompt_lens, raw_texts)` | One forward + `.generate()` under the actor adapter. Returns per-token log-probs for the generated span only (not the prompt). Parser folds into the caller. | actor-critic wrapper, obs tensor, list[str] prompts | 5-tuple | BaseVLM, torch |
| `training/distributed.py::active_adapter(ac_model, name)` | Context manager: asserts `ac_model.vlm.model.active_adapter == name` on enter, restores on exit. Enforces Inv-3 inside the trainer. | model, adapter name | ctxmgr | peft |
| `training/precision.py::Fp16State(grad_scaler)` | Wrap `torch.amp.GradScaler`. `.step(optimizer)` + `.log()` → loss-scale history. Inv-6 monitor reads history. | grad scaler | scale-factor deque | torch |
| `training/checkpoint.py::save_ppo_cot(path, ac_model, optimizer, rng, rollout_buffer_partial, wandb_run_id, step)` / `load_ppo_cot(path, ...)` | Atomic write `path.tmp/` → `fsync` → rename. Saves `{actor,critic}/` (PEFT) + `critic_head.pt` + `optimizer.pt` + `rng.pt` + `step.json` + `manifest.json` + `INTEGRITY_HASHES.txt`. iter-4 scope is a subset of master-spec §10 — full spec format lands in `J`. | path, all pieces | on-disk dir | peft, torch |
| `training/logging.py::RichDashboard(run_name, columns, console)`, `::CsvWriter(path)`, `::wandb_init(name, config)` | Three-sink logging. Rich auto-off in headless; wandb behind `--track`; CSV always. | run name, column schema | live dashboard + metrics.csv + optional wandb run | rich, pandas, wandb |
| `training/invariants.py::InvariantMonitor(checks, sample_every)` + per-Inv check funcs | Scaffold: `.maybe_run(step, ctx)` dispatches to registered checks. iter 4 calls subset synchronously at startup + end of training; runtime continuous wiring is `D`. | list of check fns, cadence | per-step invariant-status tuples | — |
| `algos/ppo_cot.py::main()` | The trainer. Uses *only* the exported APIs above. ~700 lines. | `Args` dataclass | training run → `runs/<name>/` | all of `src/cleanrl_vlm/*` |

### Active-adapter discipline (Inv-3)

Every actor forward happens inside `with active_adapter(ac_model, "actor"):`.
Every critic forward inside `with active_adapter(ac_model, "critic"):`.
Both context managers assert `ac_model.vlm.model.active_adapter == name`
on entry and **do not** set it — setting happens in
`DecoupledActorCriticVLM_COT.get_{action,value}()` itself. The ctxmgr is
a tripwire, not a setter, so a missing `set_adapter` call inside the
model class surfaces as an assertion failure rather than a silent
cross-adapter forward. Matches the spec's §8 Inv-3 binary-correctness
requirement.

## §4. PPO-COT algorithm specifics

### Log-probability of the action

An "action" in COT interface = the full generated span (think + ACTION
line). The scalar logprob per trajectory is `sum_t log p(tok_t | prefix)`
over the generated tokens only (not the prompt). The sum is stored in
the rollout buffer as `logprob_sum` — **not** the per-token log-probs,
to keep the buffer shape rank-2.

On the PPO update, we re-score the stored `full_ids` under the current
actor adapter and recompute `logprob_sum_new`; the ratio is
`exp(logprob_sum_new - logprob_sum_old)`. The PPO clip is sequence-level,
not token-level — matches spec §5 "Sequence-level ratio from summed
token log-probs across the generated span" and is the simplest correct
baseline. Token-level / advantage-weighted-per-token variants are
experimental-track material (`K-research-longhorizon`).

### Advantage + return

Standard GAE:

```text
delta_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
A_t    = delta_t + γ·λ·(1 - done_t)·A_{t+1}
R_t    = A_t + V(s_t)
```

Bootstrapped tail value from `V(s_{T})` on the final rollout obs. No
reward normalization at iter 4; reward scaling YAML-configurable (ViZDoom
VizdoomBasic returns already on a reasonable scale).

### Loss

```text
L_clip     = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
L_value    = 0.5 · (V - R)^2
L_entropy  = -E[entropy of per-token dist over generated span]    # summed, normalized by gen length
Loss       = mean(L_clip) + c_v · mean(L_value) + c_e · mean(L_entropy)
```

- `ε = 0.2`, `c_v = 0.5`, `c_e = 0.01` (Atari-tuned defaults per spec §5).
- Gradient clipping `max_norm = 0.5`.
- `update_epochs = 4`, `num_minibatches = 4`. `batch_size = num_envs ·
  num_steps`; `minibatch_size = batch_size / num_minibatches`.
- **Global-batch floor (§3).** With `num_envs = 4`, `num_steps = 32`,
  `num_minibatches = 4`, `grad_accum = 1`, `num_processes = 1` →
  `effective_batch = 128` → meets the Tier-1 floor at iter 4 single-rank.
  When we scale to multi-rank in `D`, `grad_accum` is auto-sized; the
  `InvariantMonitor` refuses to start training if `effective_batch < 128`.

### Rollout shape (iter-4 defaults)

- `num_envs = 4` parallel VizdoomBasic instances.
- `num_steps = 32` per iteration. Iteration = 4·32 = 128 env steps.
- `max_new_tokens = 256` (below spec's 512 default to save time; ablation
  later).
- Per iteration wall-clock expectation: generation is
  `num_envs × num_steps × one_generate_call ≈ 128 × ~1s ≈ 2 min` on
  2B backbone (rough; vLLM path will 5-10× this). PPO update ~30s. So
  `~2.5-3 min` per iteration on the 8-GPU node (single rank). A 100-iter
  smoke run ≈ 4-5 hours. Not feasible in a single /loop iteration —
  agent will invoke long-running training via `run_in_background` and
  schedule wakeups on milestone events (loss flatlines, curve crosses
  threshold, checkpoint lands, etc.).
- `total_timesteps = 200_000` for the first training run. Agent judges
  per §0 whether to extend or cut short.

### Precision

FP16 end-to-end (spec §3 default). `GradScaler` wraps the optimizer. Inv-6
watches scale-factor history; if scale halves repeatedly without
recovery, systematic-debugging skill is invoked.

### LoRA topology

Default groups enabled at iter 4: **text_attn + text_mlp + lm_head**. No
vision-tower LoRA on the first trainer (ablation later). `lora_r = 32`,
`lora_alpha = 64`, `lora_dropout = 0.0` (spec §14 defaults).

## §5. Prompts

Templates live at `prompts/vizdoom/basic/`; the builder reads them.

`prompts/vizdoom/basic/actor.txt`:

```text
You are playing VizdoomBasic, a simple ViZDoom scenario in which a
single monster appears randomly in front of you and you must shoot it
before a timeout.

Available actions:
- MOVE_LEFT  — strafe left
- MOVE_RIGHT — strafe right
- ATTACK     — shoot

Look at the current screen carefully. Think briefly about where the
monster is and whether you need to line up your aim. Then respond with
exactly one line of the form:

ACTION: <NAME>

where <NAME> is one of MOVE_LEFT, MOVE_RIGHT, ATTACK.
```

`prompts/vizdoom/basic/critic.txt`:

```text
You are an RL value estimator for VizdoomBasic. Given the current
screen, estimate how good this state is for a future reward of +101 for
killing the monster minus a small living cost. Respond with no text —
your hidden representation is read by a downstream regressor.
```

`prompts/vizdoom/basic/vision_probe.txt` (Inv-15):

```text
For each frame, answer three questions:
1. Is a monster visible? (yes / no)
2. Roughly where is the monster on the screen? (left / center / right / off-screen)
3. Is the player's crosshair roughly aligned with the monster's body? (yes / no / no-monster)
```

Parser (`PromptBuilder.parse_action`): split on `"ACTION:"`, strip +
upper-case the tail, lookup in the action table. On parse-fail, sample
uniformly from `action_space` (matches prototype `parse_action_cot`).
Parse-fail rate is logged as a scalar and surfaces in the rich console.

## §6. Invariants landed at iter 4

Binary-correctness invariants from spec §8 that apply to this trainer:

| Inv | What iter-4 does | Test file |
|-----|------------------|-----------|
| 1 | Assert at build + after every `set_adapter` swap that `requires_grad` is `True` iff `"lora_"` in name or param ∈ {critic_head}. | `tests/invariants/test_inv_01_lora_trainability.py` |
| 3 | `active_adapter(...)` ctxmgr wraps every actor/critic forward. Unit tests for the ctxmgr's assertions. Trainer code path spot-check via a single iteration. | `tests/invariants/test_inv_03_active_adapter.py` |
| 5 | Cross-check global grad-norm via `clip_grad_norm_` vs manual `sum(p.grad.pow(2).sum()).sqrt()` on one microbatch every 10 iterations. | `tests/invariants/test_inv_05_grad_norm.py` |
| 6 | Track `scaler.get_scale()` across a 20-iter fixture; assert no NaN/Inf in grads; on NaN: dump microbatch under `runs/<name>/nan_dumps/`. | `tests/invariants/test_inv_06_fp16_scale.py` |
| 9 | Scripted env emits rewards `[0, 1, 0, 2]`; assert advantage computation sees exactly those numbers post-scaling. | `tests/invariants/test_inv_09_reward_pipeline.py` |
| 10 | 2-episode synthetic rollout with a `done=True` at step 3; assert GAE resets — no value leakage from episode 2 into episode 1. | `tests/invariants/test_inv_10_episode_boundary.py` |
| 11 | Two rollouts seeded identically → tensor-equal. | `tests/invariants/test_inv_11_determinism.py` |
| 13 | Microbatch with extra `<pad>` + image-placeholder tokens; assert their gradient contribution is exactly zero after backward. | `tests/invariants/test_inv_13_pad_image_token_mask.py` |

Invariants **deferred** to later tasks (link in LOOP_STATE):

- Inv-2 (LoRA-weights-actually-change) → landed as a light check in this
  iter's integration test (assert weight-hash diff after 10 updates),
  promoted to a standalone invariant in `D-invariants-runtime`.
- Inv-4 (train ↔ inference logprob parity) → `E-vllm-rollout-path`
  (needs vLLM).
- Inv-7 (checkpoint round-trip) → `J-checkpoint-resume-e2e`.
- Inv-8 (image-input correctness: patch coverage, resolution floor,
  cross-rank token count) → one-shot onboarding run this iter writes
  `docs/backbone_probes/qwen3-vl-2b-instruct.md`; runtime invariant in `D`.
- Inv-12 (resume parity) → `J`.
- Inv-14 (distributed broadcast) → `D` (multi-rank).
- Inv-15 (ground-truth vision probe) → `scripts/probe_vision.py` lands
  in this iter; the `docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md`
  is generated at backbone onboarding; the **agent reviews the artifact**
  per §0 and iterates on prompts / pixel budget / resolution until the
  report reads as "model genuinely sees the scene".

## §7. Test strategy (TDD where practical)

Per §13.2 `superpowers:test-driven-development` is mandatory for new
code. Interpretation: each unit in §3's module table ships with its
test file created **before** the implementation, with a reddish
assertion, then green after impl. For units where a meaningful
failing-test-first is awkward (e.g., `BaseVLM` — loading a real VLM
inside a unit test is prohibitive), the associated test is the
integration one; impl lands with the integration test falling back from
a TDD-exact stance.

Test split (matches `tests/` layout already scaffolded):

- `tests/unit/` — CPU-safe, fast (<1s each, <10s total). Parser, factory,
  GAE, buffer, topology. Runs on every commit via CI.
- `tests/invariants/` — the eight Inv-* tests above. Some need a
  mini-model fixture (2-3 layer transformer stub so we don't load the
  real 2B backbone in unit tests); others can use the real 2B backbone
  under `@pytest.mark.gpu` + `@pytest.mark.tier1`.
- `tests/integration/test_trainer_short_run.py` — `@tier1 @gpu`, runs
  `main()` for 10 iterations on VizdoomBasic, asserts: loss finite,
  episodic return logged, LoRA weight-hash changed, checkpoint writes,
  Inv-1/3/5/6/9/10/11/13 all green.

Mini-model fixture (for invariants): a custom
`TinyVLMForImageTextToText` stub with a `text_config.hidden_size` small
enough to run on CPU + `AutoModelForImageTextToText` register. Avoids
downloading the real backbone for tests 1/3/5/10/11/13 (9 + 6 stay on
real backbone since they exercise reward / FP16 which need the real
GradScaler + real env).

## §8. Config defaults

`configs/envs/VizdoomBasic-v0.yaml`:

```yaml
tier: 1
backbone_default: Qwen/Qwen3-VL-2B-Instruct
frame_skip: 4
frame_stack:
  n: 1                         # single frame at iter 4
  layout: horizontal           # reserved for n > 1
resolution:
  width: 320
  height: 240
screen_format: RGB24           # vizdoom default
reward_clip: false             # VizdoomBasic has well-scaled rewards
max_episode_steps: 1000        # ViZDoom config default
target_score: 60.0             # living cost ~0.5/step, kill reward ~100; agent judges
```

`configs/backbones.yaml` (iter-4 entries only; future backbones appended):

```yaml
Qwen/Qwen3-VL-2B-Instruct:
  tier: 1
  auto_class: AutoModelForImageTextToText
  processor_pixel_budget:
    min_pixels: 262144         # 512*512 equivalent; per spec §4 resolution-floor
    max_pixels: 1310720        # 1280*1024 equivalent
  attn_implementation: flash_attention_2
  dtype: float16
  thinking_mode: off           # -Instruct variant
  lora_groups_default: [text_attn, text_mlp, lm_head]
  lora_rank: 32
  lora_alpha: 64

Qwen/Qwen3-VL-4B-Instruct:
  tier: 2
  auto_class: AutoModelForImageTextToText
  processor_pixel_budget:
    min_pixels: 262144
    max_pixels: 1310720
  attn_implementation: flash_attention_2
  dtype: float16
  thinking_mode: off
  lora_groups_default: [text_attn, text_mlp, lm_head]
  lora_rank: 32
  lora_alpha: 64
```

`configs/targets.yaml`:

```yaml
VizdoomBasic-v0:
  Qwen/Qwen3-VL-2B-Instruct:
    reference_score: 60.0       # agent-tuned; see AUTONOMY_LOG when updated
    zero_shot_baseline: null    # populated after H-baselines runs
    scratch_cnn_baseline: null  # populated after H-baselines runs
```

`algos/ppo_cot.py` Args (tyro dataclass) — spec §14 defaults:

```python
@dataclass
class Args:
    # Run meta
    exp_name: str = "ppo_cot"
    seed: int = 0
    run_name: str = "{exp_name}__{env_id}__{backbone_slug}__{seed}__{date}"
    track: bool = False
    wandb_project_name: str = "cleanRL-VLM"
    wandb_entity: str | None = None
    checkpoint_interval: int = 10
    checkpoint_dir: str = "runs/{run_name}/checkpoints"

    # Env
    env_id: str = "VizdoomBasic-v0"
    env_config: str = "configs/envs/{env_id}.yaml"
    num_envs: int = 4

    # Backbone
    backbone: str = "Qwen/Qwen3-VL-2B-Instruct"
    backbone_config: str = "configs/backbones.yaml"
    max_new_tokens: int = 256

    # Algo
    num_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4
    total_timesteps: int = 200_000

    # Optim
    learning_rate: float = 1e-5
    anneal_lr: bool = True

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_groups: tuple[str, ...] = ("text_attn", "text_mlp", "lm_head")

    # Distributed (iter 4 single-rank; respected when scaled in D)
    sharding: str = "deepspeed_zero2"   # or "fsdp2"
    precision: str = "fp16"             # or "bf16"
    num_processes: int = 1              # filled from accelerate
    grad_accum: int = 1                 # auto-derived at startup to hit batch floor

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
```

Filled-at-runtime fields: `batch_size = num_envs * num_steps *
num_processes`, `minibatch_size = batch_size // num_minibatches`,
`num_iterations = total_timesteps // batch_size`.

## §9. Logging

Per spec §9 "metric coverage mandate" — every scalar that might matter
later is logged **this iter**, even if unused now. Schema (`CsvWriter`
columns):

```text
global_step, iteration, total_env_steps, wall_s,
loss_total, loss_clip, loss_clip_unclipped, loss_value, loss_entropy,
approx_kl, clip_fraction, explained_variance,
grad_norm_global, loss_scale,
lr,
action_entropy_avg, action_parse_fail_rate,
ep_return_mean, ep_return_std, ep_return_min, ep_return_max, ep_return_n,
ep_length_mean, ep_length_std,
rollout_wall_s, train_wall_s, generate_wall_s,
per_env_return_0..per_env_return_{num_envs-1},
inv_1_status, inv_3_status, inv_5_status, inv_6_status,
inv_9_status, inv_10_status, inv_11_status, inv_13_status
```

Per-layer grad-norm histograms → `runs/<name>/histograms.parquet`.
Per-iteration manifest (git SHA, pip freeze, config) → `manifest.json`.

## §10. Expected deliverable sequence (→ writing-plans skill input)

This design is the input to writing-plans. That skill's plan file will
decompose the work into ~25–40 bite-sized tasks in roughly this order:

1. `configs/backbones.yaml` + `configs/targets.yaml` + `configs/envs/VizdoomBasic-v0.yaml`
2. `src/cleanrl_vlm/envs/wrappers.py` + `tests/unit/test_env_factory.py` (TDD)
3. `src/cleanrl_vlm/envs/vizdoom/action_tables.py` + `factories.py`
4. `src/cleanrl_vlm/envs/registry.py`
5. `src/cleanrl_vlm/models/lora_topology.py` + `tests/unit/test_lora_topology.py`
6. `src/cleanrl_vlm/models/heads.py`
7. `src/cleanrl_vlm/models/base_vlm.py`
8. `src/cleanrl_vlm/models/actor_critic.py` + `tests/invariants/test_inv_01_lora_trainability.py` + `test_inv_03_active_adapter.py`
9. `src/cleanrl_vlm/prompts/templates/vizdoom/basic/*.txt`
10. `src/cleanrl_vlm/prompts/builder.py` + `tests/unit/test_prompt_builder.py` + `tests/unit/test_action_parser.py`
11. `src/cleanrl_vlm/rollout/buffer.py` + `tests/unit/test_rollout_buffer.py` + `tests/unit/test_gae.py` + `test_inv_10_episode_boundary.py`
12. `src/cleanrl_vlm/rollout/in_process.py`
13. `src/cleanrl_vlm/training/distributed.py` (active_adapter ctxmgr) + `precision.py` + `test_inv_06_fp16_scale.py`
14. `src/cleanrl_vlm/training/logging.py`
15. `src/cleanrl_vlm/training/checkpoint.py`
16. `src/cleanrl_vlm/training/invariants.py` scaffold + per-Inv check funcs + the remaining invariant tests (`05`, `09`, `11`, `13`)
17. `scripts/_cluster_env.sh`
18. `algos/ppo_cot.py` — assemble everything.
19. `tests/integration/test_trainer_short_run.py` (tier1 gpu).
20. `scripts/probe_vision.py` + run on (`VizdoomBasic`, `qwen3-vl-2b`) → commit `docs/vision_probes/.../report.md`.
21. `docs/backbone_probes/qwen3-vl-2b-instruct.md` — auto-generated from the onboarding run (includes Inv-8 token-count + patch-coverage numbers).
22. Actual training run (agent kicks off via `run_in_background`, schedules wakeups on milestone events).
23. Update `docs/ALGORITHMS.md`, `docs/ENVS.md`, `docs/RECIPES.md`, `docs/RESULTS.md` (first row), `docs/BACKBONES.md` (probe link).
24. `code-reviewer` subagent on the full diff; fix findings.
25. `simplify` skill pass.
26. Commit journals; pivot LOOP_STATE to next task per priority queue.

## §11. Risks + contingencies

| Risk | Mitigation |
|---|---|
| Qwen3-VL architecture differs from spec's assumed Gated-DeltaNet / MoE → some LoRA groups no-op. | `default_target_modules` probes the actual module names via `[n for n, _ in model.named_modules()]` at startup and intersects with the requested groups. `docs/backbone_probes/...` records what actually got LoRA. |
| `AutoModelForImageTextToText` API surface may differ from prototype's `Qwen3VLForConditionalGeneration`. | `get_model_class` helper maps backbone → model class; iter 4 uses `AutoModelForImageTextToText` directly (per iter-3 smoke success). Test integrates across classes. |
| Pixel budget mismatch with vizdoom's 320×240 screen → processor upscales unnecessarily, wasting memory. | `configs/backbones.yaml` pixel budget = 262144 min, 1310720 max. `320*240 = 76800` → under min → processor upscales. Fix: either drop `min_pixels` to `320*240` or downscale in wrapper. Decision deferred to iter-4 implementation — `PromptBuilder` tests assert the final token count is finite and under a budget. |
| Generate wall-clock ≫ expected at 2B. | Fall-back to `max_new_tokens = 128`. If still too slow, prioritize promoting `E-vllm-rollout-path`. Schedule wakeups every ~30 min around the long training run rather than /loop-iterating on it. |
| LoRA-adapter swap overhead dominates rollout time. | Spec's design is one `set_adapter` call per iteration (not per step). This design respects that — generate all actions under "actor", compute values under "critic", never interleave. |
| Integration test requires GPU + model download in CI. | Test is `@tier1 @gpu`; CI job for tier1 is behind `if: false` until a self-hosted runner is wired (iter 2's `ci.yml`). iter 4 runs the integration locally before merge; CI catches regressions only once the GPU runner is live. |
| Reward scaling difference between Doom's `+101 kill / -5 step` and PPO clip window. | Leave raw; log advantage distribution (`action_entropy_avg`, `explained_variance`) and agent adjusts per §0. |
| Determinism under vizdoom + CUDA + HF generation is fragile (CUDA non-determinism, vizdoom internal RNG). | `Inv-11` test sets `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + vizdoom seed. On failure, surface which component broke; downgrade to "matches within fp16 tolerance" per §0 rather than hard-fail. |

## §12. Non-goals (explicit)

- Not a learning-curve target. §9 green = agent judges "genuinely learning."
- Not a multi-backbone sweep. Only `Qwen/Qwen3-VL-2B-Instruct` at iter 4.
- Not a paper-ablations result. That's `paper-ablations` task, triggered post-M.
- Not a "production CleanRL clone". This is a *living trainer* that the
  /loop will continue to refine per §13.3 loop cycles.

## §13. Sign-off

Self-reviewed per §13.1. Code-reviewer subagent dispatch to be run in the
commit that follows this design's landing (before first implementation
commit under §10's sequence). AUTONOMY_LOG iter-4 entry cites this file;
writing-plans skill consumes §10's sequence to emit the task-level plan.
