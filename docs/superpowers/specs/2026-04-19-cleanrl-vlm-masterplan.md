---
title: cleanrl-vlm — Master Spec
slug: cleanrl-vlm-masterplan
date: 2026-04-19
authors: Roger Creus Castanyer, Claude (brainstorming co-author)
status: draft (pending human review of written spec)
supersedes: none
---

# cleanrl-vlm — Master Spec

## Vision

**cleanrl-vlm is a CleanRL-style library + research paper for online RL finetuning of Vision-Language Models in interactive visual environments (ViZDoom, Atari, Minigrid).** It demonstrates that pretrained VLMs + efficient LoRA finetuning + a correct RL pipeline can adapt a static foundation model into an agentic one that outperforms CNN agents trained from scratch — while doubling as a reference implementation that other researchers can build on.

Development is **autonomous**: the entire library, experiments, and running documentation are produced by a Claude Code `/loop` agent operating over many cycles without human intervention, following the process rituals of the `superpowers` skillset.

---

## §0. Operating principle — "thresholds are signals, not gates"

Numerical thresholds in this spec are **starting points** to orient the /loop agent's judgment, not CI-style hard-fail cliffs.

The agent is a researcher: analyze outputs, understand failure modes, iterate on preprocessing / prompts / hyperparams / library versions, decide when work meets the *spirit* of an invariant or evidence gate. When a number fires, the correct response is **investigate → understand → iterate**, not "mark red and move on."

**Hard-fail applies only to genuine correctness bugs** — unambiguous "the code is wrong" signals:

- shape / dtype mismatches
- NaN / Inf explosions in the forward or gradient
- logprob divergence between training and rollout paths that survives investigation
- checkpoint that cannot round-trip into a fresh process
- active-adapter context violations (critic forward running under actor adapter or vice versa)
- non-deterministic behavior under a fixed seed

Everything else — tolerances, accuracies, percentages, step-count windows, target scores — is **guidance the agent tunes with evidence**. The S-1..S-12 rituals (§11), §8 invariants, §9 evidence gates, and §13 autonomy contract are framings for the agent's analysis, not rigid gates that become artificial blockers.

This principle **overrides** any numerical threshold in the sections that follow where that threshold is used for judgment rather than correctness.

---

## §1. Purpose, scope, non-goals, git strategy

### Hero paper claim

> Pretrained Vision-Language Models, with efficient LoRA finetuning and a correct on-policy RL pipeline (PPO/GRPO/RLOO), adapt to interactive visual environments (ViZDoom, Atari, Minigrid) under their native episode horizons — and outperform CNN agents trained from scratch, often while training a small fraction of parameters. Along the way, the paper contributes novel methods for long-horizon credit assignment with VLM critics.

### Ambition

- Library **and** paper, built entirely by a `/loop` autonomous agent.
- CleanRL-style clarity: single-file trainers, minimal shared abstractions, copy-paste-friendly.
- "Hybrid library" variant: single-file trainers on top of a small shared library for env wrappers, prompt builders, rollout infra, model utilities — because VLM plumbing is too heavy to duplicate per file.
- Polished enough to be the canonical reference for VLM online RL in interactive visual envs.

### In scope (v1)

- Library infra (`src/cleanrl_vlm/`): envs, models/LoRA helpers, prompts, rollout (vLLM), training (distributed, checkpointing, logging, invariants), research primitives.
- 9 canon single-file trainers: `{ppo, grpo, rloo} × {cot, action, head}` under `algos/`.
- 3 baselines: CNN-PPO from scratch, zero-shot VLM, frozen-VLM + trainable-head.
- All ViZDoom scenarios, all Atari games, all Minigrid/BabyAI envs.
- Dual-track research: `experimental/` playground with promotion ritual into `algos/`.
- Evidence dashboard (`docs/RESULTS.md`), running research journal (`docs/RESEARCH.md`), comprehensive docs site.
- Correctness invariants (§8) + standing rules (§11) + autonomous operation contract (§13).

### Out of scope (v1; reachable via spec-extend)

- LaTeX / arXiv paper writeup (the agent maintains `docs/RESULTS.md` only; the human composes the paper from it).
- Docker / cloud / cluster-orchestration tooling.
- Multi-backbone support beyond the Qwen3.5-VL family (0.8B, 4B).
- KL regularization to a reference model (default 0.0; not implemented).
- Offline algorithms (DPO, ExIt from preference data) — verifiable env rewards only.

### Hardware + constraints

- **Hardware**: 1 × node with 8 × NVIDIA RTX A6000 (48 GB each) available for all runs.
- **Atari episode horizon**: `max_episode_steps = 27_000`; **never changed** — a non-negotiable research constraint driving §5 and §6.
- **Global batch floor**: ≥ **128 samples / gradient step** for Tier-1, ≥ **256** for Tier-2. Derived at runtime from `num_gpus × per_gpu_microbatch × grad_accum`, with `per_gpu_microbatch` auto-probed at startup.
- **Dependency management**: **UV** (`pyproject.toml` + `uv.lock`).
- **Python**: 3.10+ (aligns with current `transformers` + `vllm` + hybrid-Qwen support).
- **Precision**: FP16 default (more stable for RL finetuning per empirical lore; BF16 first-class ablation).
- **License**: **MIT** for our code. Qwen3.5-VL models are Apache-2.0. All our deps are OSS-compatible with MIT.

### Git strategy

- At scaffold time, the current `master` is moved to branch **`old`** (preserved, frozen, never touched again).
- New development lands on `master` starting from a scaffold PR that creates the full skeleton described in §2.
- Scaffold PR also rewrites `CLAUDE.md` to carry §13 verbatim at its top (instruction-priority override, see Appendix B).
- No force-push to `master` under any circumstance. No deletion of `old`.

---

## §2. Repo layout

```
cleanrl-vlm/
├── pyproject.toml  uv.lock  LICENSE (MIT)  mkdocs.yml  CLAUDE.md  CHANGELOG.md
├── README.md                         # 1-page pitch + quickstart
├── src/cleanrl_vlm/                  # shared library (hybrid layer)
│   ├── envs/                         # gymnasium factories + wrappers (vizdoom/atari/minigrid)
│   │   ├── registry.py               # single point of env registration
│   │   ├── vizdoom/  atari/  minigrid/
│   │   ├── wrappers.py               # FrameStack, DiscreteMultiBinaryWrapper, ScreenOnly, ...
│   │   └── vizdoom_action_tables.py  # button-name tables per scenario
│   ├── models/
│   │   ├── base_vlm.py               # backbone-agnostic wrapper
│   │   ├── lora_topology.py          # configurable target-module groups + auto-probe
│   │   ├── heads.py                  # CriticHead, ActorHead
│   │   └── backbones/                # Qwen3.5-VL adapters (processor, tokenizer, vision config)
│   ├── prompts/
│   │   ├── builder.py                # prompt assembly + chat template + image tokens
│   │   └── templates/<env_family>/   # COT / action / head prompt templates
│   ├── rollout/
│   │   ├── vllm_server.py            # vLLM worker, multi-LoRA serving, adapter sync
│   │   ├── buffer.py                 # rollout buffer + GAE
│   │   └── in_process.py             # HF generation path (non-vLLM)
│   ├── training/
│   │   ├── distributed.py            # accelerate + DS/FSDP plumbing
│   │   ├── precision.py              # FP16/BF16 config, GradScaler watcher
│   │   ├── checkpoint.py             # §10 save/load/round-trip
│   │   ├── logging.py                # Rich + wandb + CSV stack
│   │   ├── invariants.py             # §8 InvariantMonitor + each Inv-*
│   │   └── loop_state.py             # LOOP_STATE / AUTONOMY_LOG helpers
│   └── research/                     # primitives for experimental algos
├── algos/                            # curated canon — 9 single-file trainers
│   ├── ppo_cot.py    ppo_action.py    ppo_head.py
│   ├── grpo_cot.py   grpo_action.py   grpo_head.py
│   └── rloo_cot.py   rloo_action.py   rloo_head.py
├── experimental/                     # playground (§6)
│   └── <method-slug>/
│       ├── MOTIVATION.md
│       └── trainer.py
├── baselines/
│   ├── cnn_ppo.py           # non-VLM PPO from scratch (from current src/ppo.py, polished)
│   ├── zero_shot_vlm.py     # prompt-only, no finetuning
│   └── frozen_vlm_head.py   # frozen VLM + trainable MLP action head (no LoRA on VLM)
├── configs/
│   ├── targets.yaml                  # reference target scores per env (Tier-1 and Tier-2)
│   ├── backbones.yaml                # per-backbone defaults (pixel budget, attn impl)
│   └── envs/<env_id>.yaml            # per-env hyperparameters (frozen per Q10=A)
├── prompts/<env>/                    # human-facing COT / action / head prompt authoring
├── scripts/
│   ├── loop_step.py                  # single /loop iteration entrypoint
│   ├── probe_vision.py               # §4 / Inv-15 vision-perception probe
│   ├── run_sweep.py                  # launch Tier-2 sweeps
│   ├── resume.py                     # §10 resume with round-trip + parity checks
│   ├── build_dashboard.py            # regenerate docs/RESULTS.md
│   └── doc_audit.py                  # docs-comprehensive-pass runner
├── docs/
│   ├── index.md                      # mkdocs landing
│   ├── ARCHITECTURE.md
│   ├── ALGORITHMS.md
│   ├── ENVS.md
│   ├── BACKBONES.md
│   ├── RECIPES.md
│   ├── RESULTS.md                    # live, agent-maintained
│   ├── RESEARCH.md                   # live, research journal
│   ├── INVARIANTS.md
│   ├── CHECKPOINTING.md
│   ├── LOGGING.md
│   ├── CONTRIBUTING.md
│   ├── TROUBLESHOOTING.md
│   ├── backbone_probes/<name>.md     # auto-generated
│   ├── vision_probes/<env>_<backbone>/report.md   # auto-generated
│   └── superpowers/
│       └── specs/
│           ├── 2026-04-19-cleanrl-vlm-masterplan.md   # THIS FILE
│           ├── amendments/
│           │   └── YYYY-MM-DD-<topic>.md
│           └── plans/
│               └── YYYY-MM-DD-<slug>.md              # writing-plans output
├── tests/
│   ├── invariants/                   # test_inv_01.py .. test_inv_15.py
│   ├── unit/                         # pure-python, CPU-safe
│   ├── integration/                  # cross-module, may need 1 GPU
│   ├── smoke/                        # Tier-1 end-to-end (0.8B backbone, fast)
│   └── soak/                         # Tier-2 longer runs (manual trigger)
├── LOOP_STATE.md                     # next-task pointer, per-combo status
├── AUTONOMY_LOG.md                   # append-only journal of decisions
└── .github/workflows/                # CI: lint, type-check, unit, Tier-1 smoke; docs deploy
```

---

## §3. Model & training stack

### Backbones

- `Qwen/Qwen3.5-VL-0.8B` — Tier-1 smoke + debug. Hybrid Gated-DeltaNet + Gated-Attention + sparse MoE; 24 layers; hidden 1024; 262K context; thinking mode **off** by default (prone to thinking loops — agent opts in with a loop-detector guardrail).
- `Qwen/Qwen3.5-VL-4B` — Tier-2 paper runs. 32 layers; hidden 2560; 262K context; **native thinking mode on by default** (use this directly for the COT interface; removes the need to engineer `THINK:` / `ACTION:` prompt scaffolding).
- Loaded via `AutoModelForCausalLM` + `AutoProcessor` (verify at onboarding time).
- Requires a recent `transformers` install (likely from git at scaffold time — web-fetch latest support notes per S-5).
- Apache-2.0 license.
- Both have MTP (multi-token prediction); vLLM speculative decoding enabled by default.

### Precision

- **FP16** is the default for all training runs (user direction — empirically more stable than BF16 for RL finetuning of LLMs).
- **BF16** is the first-class ablation; trainers accept a `--precision {fp16,bf16}` flag.
- Loss scaling via `accelerate`'s `GradScaler`; scale-factor history watched by Inv-6 (§8).

### Distributed infrastructure

- **Orchestrator**: `accelerate`.
- **Default sharding**: DeepSpeed **ZeRO-2** (matches the existing repo config; mature MoE handling; easy ZeRO-3 fallback for 8B+ backbones later via spec-extend).
- **Alternative sharding**: **FSDP2** behind a `--sharding {deepspeed_zero2,fsdp2}` flag, supported from day one for ablation purposes.
- **Gradient checkpointing**: on by default at 4B.
- **Offloading** (optimizer / param / activation): off by default; enabled per-config when an OOM is observed.
- **Attention**:
  - Flash-attn v2 on Gated-Attention layers.
  - DeltaNet is linear-attention — no flash kernel needed.
  - SDPA (PyTorch) fallback for environments where flash-attn import fails; the fallback triggers a CI warning but not a failure (research nodes with mismatched CUDA can still run).
  - Import-time sanity: run a 1-step microbatch forward and assert no NaN/Inf to verify the kernel plumbing is correct *before* any training starts.
- **Per-GPU microbatch auto-probe**: at startup, try microbatch size `s` on 1 forward+backward; double until OOM; back off one step; record in `runs/<name>/microbatch_probe.json`. Used to derive `grad_accum` to hit the global-batch floor.
- **External-library research rule** (per S-5): any change to the sharding strategy, precision handling, attention kernel, or accelerate/DS version triggers a web-fetch of current docs first; the summary lands in the PR body and in `AUTONOMY_LOG.md`.

### Global batch floor

VLMs need **large batches** to learn. Effective batch **per gradient step** ≥ 128 (Tier-1) / ≥ 256 (Tier-2). Computed as `num_gpus × per_gpu_microbatch × grad_accum`. Rollout length (`num_envs × num_steps`) auto-sized so each iteration produces at least one gradient step's worth of data at the target global batch. Invariant: at training start, `effective_batch >= floor`, else refuse to start.

### LoRA topology — fully configurable

- Dual-adapter architecture: one `actor` LoRA and one `critic` LoRA on the **same** base VLM weights (matches the current repo). Adapter is swapped via `self.vlm.model.set_adapter(name)` for each forward.
- Target modules exposed as **group-level YAML booleans** so the paper can ablate "where does LoRA help most":
  - `text_attn` — text-tower attention projections (q, k, v, o + Gated-Attention projections + DeltaNet input/output projections)
  - `text_mlp` — text-tower FFN (gate, up, down)
  - `text_moe` — per-expert FFN params (investigate whether PEFT handles per-expert LoRA cleanly; fallback: LoRA on the router if not)
  - `vision_attn` — vision tower attention
  - `vision_mlp` — vision tower FFN
  - `merger` — vision-to-text projection layer
  - `lm_head` — output embedding / LM head
- **Default set = largest tractable subset**, discovered at startup: enable groups in a fixed priority order (text_attn, text_mlp, lm_head, vision_attn, vision_mlp, merger, text_moe) adding one group at a time; run 1 microbatch forward+backward; retain the widest group set that succeeded without OOM. Recorded in `runs/<name>/lora_topology_probe.json`.
- **Group-enable ablation** is first-class in the paper.

### KL regularization

- **Default 0.0**; **not implemented** as a feature in canon trainers.
- No reference model kept, no base-logprob cache.
- Rationale (per user direction): this library targets verifiable-reward domains (env returns), where KL-to-base is not standard. The simplicity win is substantial (no extra forward pass, no ref weights in memory, no KL coef tuning).
- Re-introducible via spec-extend if later research argues it helps long-horizon credit assignment.

### Inference (rollouts)

- **vLLM** serves the `actor` LoRA for COT rollouts (batch generation across parallel envs).
- Trainer → vLLM adapter sync each iteration (LoRA weights pushed to server).
- Use vLLM's multi-LoRA serving (`LoRARequest`), PagedAttention, dynamic batching, MTP speculative decoding.
- Per S-5: the /loop agent web-fetches current vLLM docs before every change involving vLLM (LoRA hotswap API, multi-adapter serving, precision inheritance, speculative-decoding config). Summary → `AUTONOMY_LOG.md`.
- Non-generation interfaces (**action-scoring**, **MLP-head**) skip vLLM and run a single in-process forward pass — no generation loop needed.

### Checkpointing (summary — full spec in §10)

Save everything needed for bit-exact resume: LoRA adapters, heads, optimizer state, GradScaler state, scheduler, RNG (torch/numpy/python/per-GPU CUDA), env state per rank, partial rollout buffer, wandb run id, CSV offset, metric history, config snapshot, manifest. Round-trip test on every checkpoint (Inv-7). Atomic write. Integrity hashes. Retention policy. SIGTERM handler flushes within 60 s.

---

## §4. Environments & observation correctness

### Coverage

- **ViZDoom**: all scenarios in `vizdoom.gymnasium_wrapper` — basic, corridor (deadly_corridor), defend_center, defend_line, health_gathering, my_way_home, predict_position, take_cover, deathmatch.
- **Atari**: full ALE gymnasium suite (~57 games).
- **Minigrid / BabyAI**: the full gymnasium-registered grid suite.
- Single point of env registration: `src/cleanrl_vlm/envs/registry.py`.
- Adding an env = (1) registry entry, (2) prompt template under `prompts/<env_family>/`, (3) per-env config YAML under `configs/envs/`, (4) target-score entry in `configs/targets.yaml`, (5) patch-coverage probe (§4.3) + Inv-15 vision probe, (6) Tier-1 or Tier-2 tier assignment.

### Atari horizon

`max_episode_steps = 27_000` is **fixed**. Any future env-wrapper that would cap earlier must be opt-in per-run, never a default. This is *the* research constraint driving §5 novel-methods work.

### Frame stacking

- Atari / ViZDoom: default **4-frame stack**, rendered as a single **horizontally-tiled** image passed to the VLM in one call, with a text note `"four game frames in order t-3, t-2, t-1, t (left to right)"`. Preserves temporal signal inside one VLM forward.
- Minigrid: **single frame** default (state is already symbolic-rich; stacking hurts more than helps).
- Per-env YAML controls: `frame_stack: {n: <int>, layout: horizontal|grid|per-call-sequence}`. First-class ablation.

### Action wrappers

- `DiscreteMultiBinaryWrapper(button_names)` factory generalizes the existing `DeadlyCorridor` / `DefendTheLine` wrappers. One button-name table per vizdoom scenario in `envs/vizdoom_action_tables.py`.
- Atari uses standard ALE wrappers + `NoopReset`, `EpisodicLife` (disabled by default — the ALE full lifetime is the paper-standard horizon), `FireReset`, `ClipReward` (configurable).
- Minigrid discrete actions are native.

### Preprocessing

Per-env YAML: resize target, grayscale vs RGB, frameskip, sticky actions, noop-reset, episodic-life, reward clipping. Atari wrappers are kept from the existing repo but audited via Inv-8/9/10.

### Observation correctness probes (core to "make sure the model actually sees")

Every new `(env, backbone)` pair ships with three probes, run at onboarding time and re-run whenever preprocessing / resolution / frame-stacking or backbone change:

1. **Patch-coverage probe** — synthesize a known image (colored quadrants + readable text labels); run through the processor; assert image tokens are present in `input_ids` and their count matches `(H/patch_h) × (W/patch_w)` for the backbone's vision config. Zero image tokens → hard fail (correctness bug).
2. **Resolution-floor probe** — compare the raw env frame shape vs the processor's `min_pixels` / `max_pixels` budget; assert no silent downscale below a per-env floor (default: ≥ half of the raw resolution on each dimension unless the env YAML opts out).
3. **Inv-15: Ground-truth vision probe** — §8 content. Summary here:
   - Script `scripts/probe_vision.py <env> <backbone>` rolls a scripted deterministic episode (20 frames), extracts programmatic ground truth from env state (vizdoom labels buffer, atari RAM maps where documented, minigrid symbolic grid), asks the VLM a fixed battery of env-specific questions (prompts in `prompts/<env>/vision_probe.txt`), and produces an artifact bundle at `docs/vision_probes/<env>_<backbone>/report.md` with embedded frames + questions + answers + ground truth.
   - **Per §0**: the probe is a **signal** for the agent's judgment, not a numerical cliff. The agent reviews the report; if perception looks poor or failure patterns cluster (e.g., HUD-reading fails while spatial questions pass), it iterates — bump resolution, try different frame-stack layout, toggle thinking mode, adjust patch size, reword prompt — and re-runs until its own analysis reads as "the model genuinely sees the scene."
   - Re-runs on: new backbone onboarding, new env onboarding, image-preprocessing change, frame-stacking change, resolution change, `transformers` or vLLM version bump, weekly during long-running Tier-2 campaigns.

### Tiering

- **Tier-1** (CI fast path, gates PRs): runs on the 0.8B backbone, ≤ 10 min, 3 envs — `VizdoomBasic-v0`, `ALE/Pong-v5`, `MiniGrid-Empty-5x5-v0`. Smoke-confirms the end-to-end stack.
- **Tier-2** (overnight / manual / paper runs): everything else, on the 4B backbone.
- Per-env YAML has a `tier: 1|2` field.

---

## §5. Algorithms, interfaces, long-horizon credit assignment

### Canon (9 single-file trainers)

`algos/{ppo, grpo, rloo}_{cot, action, head}.py`. Each ~500–800 lines (CleanRL-like, but heavier because VLM plumbing). Each trainer is self-contained for reading but calls into `src/cleanrl_vlm/` for reused primitives (rollout buffer, logging, checkpointing, invariants).

### Interfaces

- **COT** (`cot`) — **hero**. VLM generates free-form text under the native `<think>...</think>` mode. Action parsed out of a trailing `ACTION: <NAME>` token or — for backbones with native thinking — parsed from the structured response after the closing `</think>` tag. Token-level log-probs over the generated span feed the RL loss.
- **Action-scoring** (`action`) — ablation. Score each integer-token `"0".."N-1"` appended to the prompt; sample from those log-probs; no generation loop.
- **MLP-head** (`head`) — ablation. Drop an MLP on the VLM's last non-pad hidden state; outputs action logits directly; no LM head involvement.

All Tier-2 paper runs use COT across all envs and all three algos. Action-scoring and MLP-head run on a representative env subset (2 vizdoom + 2 atari + 2 minigrid) to establish the ablation evidence that COT is the right choice.

### Algorithms

- **PPO-COT** (default flagship). Standard clipped surrogate. Sequence-level ratio from summed token log-probs across the generated span. GAE(λ) on env rewards using a VLM critic (critic LoRA adapter on same base weights). Atari-tuned defaults: clip 0.2, γ 0.99, λ 0.95. Ref implementations of the CNN variant live in `baselines/cnn_ppo.py`.
- **GRPO-COT**. No learned value; group-relative advantage — for each state, K (default 4–8) full COT rollouts, advantage is centered reward within the group. **Long-horizon incompatibility made explicit**: with Atari's 27K cap, K full rollouts per state is infeasible. Canon GRPO-COT ships with a pragmatic default (sub-trajectory grouping; fragment episodes into K-step slices, apply GRPO per slice with bootstrapped tail value) and the more principled fixes live under `experimental/` until promoted.
- **RLOO-COT**. K trajectories per state; each trajectory's baseline is the mean of the other K − 1. Same long-horizon issue as GRPO; same pragmatic default.
- **Action-scoring and MLP-head variants** of PPO/GRPO/RLOO run the same algorithm templates but with the simpler action distribution. No generation loop → faster; lower research interest (VLM-as-visual-encoder story).

### Long-horizon credit assignment (research mandate; see also §6)

Atari's 27K horizon combined with a 4B VLM critic makes pure Monte-Carlo returns prohibitively expensive in wall-clock + variance. The /loop agent MUST treat long-horizon credit assignment as a research contribution. Seed research directions (each belongs in `experimental/` with a `MOTIVATION.md` per §6):

1. **Scalable VLM critics** — token-level value regression on the `<think>` span; asymmetric critic (critic sees a short state-digest text, actor sees the full VLM output); bootstrapped TD(λ); V-trace for off-policy correction when vLLM and trainer drift between adapter syncs.
2. **Group-baseline replacements for GRPO/RLOO** — sub-trajectory GRPO with bootstrapped tail value (the pragmatic canon default); Monte-Carlo / TD blend `(1−α)·MC + α·TD` with a schedule; hindsight subgoal labeling via the VLM itself (LLM proposes subgoals → dense shaped rewards).
3. **Efficient credit assignment** — Retrace, Emphatic TD, V-trace, attention-over-trajectory critics; Decision-Transformer-style reward-conditioned rollouts as an alternative head family under `experimental/`.
4. **VLM-as-reward-model** — the frozen base VLM as a zero-shot dense reward rater; hybrid sparse-env + dense-VLM rewards with a configurable mix (first investigate "does this help or hurt?").

The /loop agent is expected to originate additional directions and to *genuinely pursue* these — this section is not a menu to tick off; it is a starting set. Novel findings (positive or negative) land in `docs/RESEARCH.md` per §6.

---

## §6. Research track (dual-track protocol)

The `/loop` agent is a researcher, not only a builder. Between canon-stability cycles, it proposes, implements, and tests novel methods in `experimental/` and promotes or parks them with evidence.

### Promotion protocol

1. **Propose.** Agent picks an open question (starts from §5 seeds; grows via `docs/RESEARCH.md`). Writes `experimental/<method-slug>/MOTIVATION.md`:
   - Hypothesis (1 paragraph).
   - Why this might help long-horizon VLM RL (1 paragraph).
   - Concrete success criterion — *the agent's own yardstick for its own judgment* per §0. Example: "learns faster than PPO-COT on VizdoomCorridor across 3 seeds, by an amount the agent judges meaningful given seed variance."
   - Compute budget cap.
2. **Implement.** `experimental/<method-slug>/trainer.py` — forked from the closest canon trainer, minimal delta. Reuses `src/cleanrl_vlm/` primitives.
3. **Run.** Same invariant coverage as canon (§8). Short-circuit early if obviously failing: reward collapse, NaN, no learning signal after ~10 % of budget, logprob drift beyond Inv-4 investigation tolerance.
4. **Decide.**
   - **Promote** — criterion met on the envs the agent chose to target (multiple, ideally diverse): `experimental/<slug>/trainer.py` → `algos/<slug>.py`; entry in `docs/RESULTS.md`; paragraph in `docs/ALGORITHMS.md`; `docs/RESEARCH.md` appended with PROMOTED tag, run IDs, plots.
   - **Park** — "tried, didn't help": `MOTIVATION.md` gets a `POSTMORTEM` section (what happened, likely cause, lesson for future attempts). `trainer.py` stays, tagged `inactive` (don't re-run, don't delete — institutional memory). `docs/RESEARCH.md` appended with PARKED tag.
5. **Journal.** Every promote/park appends to `docs/RESEARCH.md` with date, run IDs, key plots, lessons.
6. **Exploration scheduling.** Loop alternates: **~3 cycles of canon hardening** (smoke tests, re-runs, flaky fixes, doc improvements) per **1 cycle of research exploration**. Ratio is the agent's to tune based on where the work is bottlenecked.
7. **Anti-novelty guard.** Before proposing method M, agent searches `experimental/` + `docs/RESEARCH.md` for prior similar attempts. If found, either revives with a principled change (documented in MOTIVATION.md) or moves on.

---

## §7. Baselines

Three first-class baselines, each with its own file under `baselines/`:

- `baselines/cnn_ppo.py` — non-VLM PPO from scratch (polished descendant of the existing `src/ppo.py`; CleanRL-style CNN encoder; the reference point for the hero claim).
- `baselines/zero_shot_vlm.py` — pure prompting, no RL. Establishes that pretraining alone does not solve the env.
- `baselines/frozen_vlm_head.py` — frozen VLM + trainable MLP action head (no LoRA on the VLM weights). Ablates whether LoRA matters.

Every baseline is a first-class runnable script and ships an evidence row in `docs/RESULTS.md` once its Tier-2 runs complete. Baselines honor all §8 invariants that apply to their architecture (e.g., zero-shot-VLM doesn't train, so only Inv-8/15 image-correctness invariants apply; frozen-VLM-head must show non-LoRA params are actually frozen).

---

## §8. Correctness invariants

Every item below is an **automatic test**, not a comment in code. Canon trainers assert the relevant invariants at startup before any gradient step; long-running trainers sample them via `InvariantMonitor` at agent-chosen intervals.

**Applying §0**: binary "the code is wrong" invariants stay binary; tolerance-based invariants have starting numbers the agent tunes with evidence.

- **Inv-1 — LoRA trainability split.** `requires_grad=True` iff `"lora_" in name` or param belongs to a head (`critic_head`, `actor_head`). Others `False`. Asserted at build + periodically. Binary.
- **Inv-2 — LoRA actually changes.** Weight hash over a training interval the agent considers sufficient (e.g., O(hundreds) of steps) must differ for every LoRA tensor; non-LoRA weight hashes must be identical to step 0. Dead-adapter detection (weights never move) → investigate.
- **Inv-3 — Active adapter sanity.** Before each actor forward, `active_adapter == "actor"`; before each critic forward, `== "critic"`. Wrap forwards in a context manager that asserts + restores. Binary.
- **Inv-4 — Training ↔ inference logprob parity** *(the most important silent-bug guard; per §0 the starting threshold is a guide, not a cliff).* Every agent-chosen K iterations (start with K = 50), sample a handful of cached rollout triples `(obs, prompt, generated_ids)` and re-score them via BOTH the vLLM serving path AND the HF+PEFT training forward. Starting tolerance: `|Δlogprob_per_token|` around 1e-3 mean. Drift above that → investigate (tokenizer, mask, image preprocessing, stale adapter on vLLM server, padding, precision difference, attention impl). Drift that survives investigation → real correctness bug (hard fail).
- **Inv-5 — Gradient-norm sanity.** Log per-layer + global grad-norm every step. Cross-check: once every agent-chosen N iterations, compute global norm via `clip_grad_norm_` AND an independent `sum(p.grad.pow(2).sum() ...).sqrt()` on one batch; assert equal up to fp16 tolerance. Binary (either the two computations agree or there's a real bug).
- **Inv-6 — FP16 stability.** Track `GradScaler` scale factor. Repeated halving without recovery → investigate (architecture numerical issue, LR too high, bad init). NaN/Inf in gradients → first occurrence dumps full microbatch to `runs/<name>/nan_dumps/` for debugging.
- **Inv-7 — Checkpoint round-trip.** Every checkpoint: save → spawn fresh process → load → deterministic rollout with fixed seed → assert logprobs match pre-save rollout within `±1e-5`. Optimizer state round-trips (same param → same momentum). Binary.
- **Inv-8 — Image-input correctness.** §4 patch-coverage + resolution-floor probes re-run as CI tests for every `(env, backbone)`. Runtime: for a fixed dummy obs, the processor must produce the same token count across GPUs and across ranks (multimodal DDP divergence hides here). Binary.
- **Inv-9 — Reward-pipeline integrity.** Synthetic env with scripted rewards `[0, 1, 0, 2]`; assert advantage computation sees exactly those numbers (post scaling/clipping toggles). Catches reward-scale bugs silently zero-ing the signal. Binary.
- **Inv-10 — Episode-boundary masking.** Synthetic 2-episode rollout; assert GAE λ resets exactly at `done=True` boundaries (no value leakage). Binary.
- **Inv-11 — Determinism under fixed seed.** `make_deterministic(seed) → rollout_A`; again `rollout_B`; must be bitwise equal. Binary.
- **Inv-12 — Resume parity.** Continuous training of 200 steps → metric log A. Training of 100 steps, checkpoint, restart-from-checkpoint, 100 more steps → metric log B. Assert A ≈ B within fp16 tolerance. Binary in the aggregate-metric sense; the agent judges what "within tolerance" means for the particular metric.
- **Inv-13 — Padding & image-token masking.** Inject extra `<pad>` and image-placeholder tokens in a microbatch; confirm their gradient contribution is exactly zero. Binary.
- **Inv-14 — Distributed-broadcast agreement.** After `accelerator.backward` + `step`, hash a sample of LoRA weights on rank 0 vs rank N-1; assert identical. Catches silent ZeRO/FSDP divergence. Binary.
- **Inv-15 — Ground-truth vision probe.** §4 procedure. **Signal, not gate**: agent reviews the artifact report; iterates on preprocessing / prompts / resolution / thinking-mode / patch-size / frame-stack layout / backbone version until the artifact reads as "the VLM genuinely sees the scene." Hard-fail only on the programmatic impossibilities (zero image tokens, processor crash, model output non-parseable on every frame) — those are real bugs.

Each invariant ships as a pytest test in `tests/invariants/` AND wires into a runtime `InvariantMonitor`. On CI: any binary invariant failing → red. In long Tier-2 runs: failure triggers auto-checkpoint + loud log + stop-that-one-run (keep the /loop going with a different task).

---

## §9. Evidence, autonomy (in-run), logging, CI

### Evidence model ("green" criteria)

A `(env, algo, interface, backbone)` combo is **green** when the agent judges all of the following — **using §0 throughout** — to have been established:

- **Target-score reference.** Mean episodic return over the last 10 % of training sits at a level **consistent with genuine learning** on the env. Targets in `configs/targets.yaml` are **reference points for the agent's judgment**, seeded from existing curves + published baselines where available, tuned by the agent via spec-extend when its own evidence establishes better/worse plausible ceilings.
- **Health gates.** Inv-4, Inv-5, Inv-6, Inv-15 stay in acceptable territory throughout the run (agent judges "acceptable" per §0). No NaN events. Reward trend isn't pathologically collapsing. Grad-norm band is consistent.
- **Seed consistency.** Tier-2 runs use ≥ 3 seeds; seed-mean + band is what the agent assesses (not "within ±3 %" — that's guidance, not a cliff).
- **Reproducibility.** A re-run with the same seed + config produces results the agent judges reproducible.

Dashboard — `scripts/build_dashboard.py` → `docs/RESULTS.md`: per-combo status (green / yellow / red — with a paragraph justification the agent writes), learning curves (seed-mean ± band), Tier-2 tables, ablation summaries. Regenerated after every Tier-2 run.

### In-run autonomy (distinct from the /loop contract in §13)

Within a single training run (not the outer /loop), the trainer itself is autonomous too:

- Detects OOM → reduce per-gpu microbatch, bump grad-accum, continue (log it).
- Detects logprob drift → re-sync adapter, re-run Inv-4, continue (log it).
- Detects checkpoint round-trip failure → **hard fail** (correctness bug).
- Detects env instability (infinite hang, crash loop) → kill that env worker, continue with the rest, log it.

### Logging stack

- **Rich console** (`rich` library). Live colored dashboard during a run: global step, per-env episodic return, grad norm, loss scale, vLLM cache hit rate, invariants status, ETA. Auto-off in headless / CI.
- **Weights & Biases** (opt-in via `--track`). Mirrors every scalar + histograms + videos (`wandb.Video`) from eval episodes every N iterations.
- **CSV** (always on). `runs/<name>/metrics.csv` — one row per step with every scalar, timestamped. `runs/<name>/histograms.parquet` for distributions (per-layer grad norm, LoRA weight norms, token-length distribution).
- **Metric coverage mandate**: every quantity the paper or the agent might want later gets logged **now**. Examples: loss (clipped, unclipped), approx KL (even though KL coef is 0), entropy, explained variance, clip fraction, grad norm (global + per-layer), loss scale, LR, action distribution entropy, episodic return/length (mean/std/min/max), rollout wall-time, train wall-time, vLLM generation latency, adapter-sync wall-time, per-env return, invariant status, ...
- **Run manifest**: `runs/<name>/manifest.json` — config + git SHA + env versions + GPU info + pip freeze — needed for evidence audit.

### CI

`.github/workflows/ci.yml`:

- **ruff** lint + format check.
- **pyright** (or equivalent strict type-checker) on `src/cleanrl_vlm/`.
- **pytest** — unit + invariant tests (the CPU-safe subset).
- **Tier-1 smoke** — on a self-hosted GPU runner (required): `pytest -m tier1` on 0.8B backbone, 3 envs, ≤ 10 min wall-clock.
- **Docs build** — `mkdocs build --strict` to catch broken links and missing references.
- Pre-commit hooks mirror the CPU subset.
- Required status checks on `master`; `old` is frozen + untouched.
- Docs deployment: `.github/workflows/docs.yml` builds + deploys to GitHub Pages on push to `master`.

---

## §10. Checkpoint & resume (save everything, resume bit-exact)

### Directory format (atomic write + rename + integrity manifest)

```
runs/<name>/checkpoints/step_XXXXXX/
├── model/
│   ├── lora_adapters/{actor, critic}/   # PEFT save_pretrained
│   ├── critic_head.pt
│   ├── actor_head.pt                    # if head interface
│   └── config.json                      # backbone, topology, precision
├── optimizer/
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── grad_scaler.pt
│   └── zero_state.pt                    # DS/FSDP sharded state if sharded
├── training/
│   ├── step.json                        # global_step, iteration, total_env_steps, wall_s
│   ├── rng.pt                           # torch, numpy, python-random, per-GPU CUDA RNG
│   ├── rollout_buffer.pt                # partial rollout if interrupted mid-iter
│   └── invariants_snapshot.json         # last observed values of Inv-1..Inv-15
├── envs/
│   └── env_state_rank{N}.pt             # vizdoom savefiles, atari RAM+RNG, minigrid grid+RNG
├── logging/
│   ├── wandb_run_id.txt                 # wandb.init(id=..., resume="allow")
│   ├── csv_offset.json                  # byte offset + row count for append
│   └── metric_history.parquet           # full metric timeline so far
├── config.yaml                          # effective config post-auto-probe
├── manifest.json                        # git SHA, pip freeze, GPU info, versions
└── INTEGRITY_HASHES.txt
```

Write path: `step_XXXXXX.tmp/` → fsync → rename to final name → update `latest` symlink. Partial writes leave the previous `latest` untouched.

### Retention policy

Keep: last 3 + every 10th + first + last-known-good (most recent that passed Inv-7). Others GC'd.

### SIGTERM

60 s budget to flush a final "interrupt" checkpoint. `scripts/resume.py` picks it up transparently.

### Resume CLI

- `--resume auto` — default; pick `latest` symlink if present, else fresh.
- `--resume <path>` — explicit path.
- `--resume none` — force fresh run.

### Resume gate

`scripts/resume.py` runs Inv-7 (round-trip) and Inv-12 (resume parity) against the loaded state before continuing training. Failure of either → refuse to resume, write a diagnostic bundle, raise. (The only case the /loop is allowed to spawn a fresh run-name instead of resuming an interrupted one.)

### wandb resume

`wandb.init(id=saved_id, resume="allow")`. CSV resume appends from saved byte offset. Rich console replays the last 50 lines from metric_history before going live.

### Preemption test (CI)

A test that sends SIGTERM at a random step during Tier-1 smoke, asserts checkpoint landed, asserts a resumed run's metrics match a continuous run within fp16 tolerance.

---

## §11. Standing rules for ongoing development (perpetual correctness)

The spec is a **living contract**. Every feature the /loop agent adds — canon trainer, experimental method, new env, new backbone, perf optimization, docs change — must respect the following rules **always**. Violations block merge via CI.

- **S-1 — Correctness coverage never decreases.** Each new `algos/*.py`, `experimental/*/trainer.py`, `envs/*`, backbone registration, or invariant-relevant change must ship with tests **in the same PR** exercising the applicable §8 invariants. CI enforces via file-pattern rule (e.g., "any new file under `algos/` without a matching `tests/algos/test_<name>.py` → red").
- **S-2 — New failure modes → new invariants.** If a feature introduces a novel failure class (new attention kernel, new reward-mixing scheme, new sharding mode, new quantization path), the agent MUST draft a new `tests/invariants/test_<inv>.py`, append to §8 via spec-extend, AND wire it into `InvariantMonitor`.
- **S-3 — No perf optimization without parity.** Any change pitched as "faster" (kernel swap, precision change, batching scheme, vLLM version bump, etc.) must prove **logprob parity (Inv-4)** against the pre-change path before merge. If numerics change materially, it is a *new method* in `experimental/`, not a drop-in replacement.
- **S-4 — Max-effort rule.** Never "just get it working" for a merge. Implement the thorough version first; if a fast-path is needed for compute, add it *alongside* the reference path (which becomes the regression oracle). Shortcuts become next-cycle tech debt — and the /loop is the one paying it.
- **S-5 — External-library research before feature.** Every feature touching `vllm`, `accelerate`, `deepspeed`, `peft`, `transformers`, `flash_attn`, `rich`, `wandb`, `torch.compile`, etc. — web-fetch current docs first, summarize in the PR description + `AUTONOMY_LOG.md`. Forever; not one-time.
- **S-6 — Backbone onboarding ritual.** New backbone = registry entry + tokenizer/processor probe + §4 image-input probes + Inv-1..Inv-14 pass on Tier-1 smoke + `docs/backbone_probes/<name>.md`. No backbone ships without all six.
- **S-7 — Env onboarding ritual.** New env = factory + wrappers + prompt template + per-env config YAML + target-score entry + patch-coverage probe + Tier-1 smoke green on 0.8B. No env ships without all seven.
- **S-8 — Algorithm onboarding ritual.** New algorithm (canon or experimental → promoted) = trainer file + tests covering Inv-4/5/10/11 + `docs/ALGORITHMS.md` paragraph + Tier-2 "green by agent judgment" on a diverse env subset + `docs/RESULTS.md` entry.
- **S-9 — Research promotion ritual.** `experimental/ → algos/` promotion = MOTIVATION.md success criterion met + ≥ 3 env greens + Inv-1..Inv-14 pass + `docs/RESEARCH.md` PROMOTED entry with run IDs and plots.
- **S-10 — Flexibility within the floor.** These rituals are the floor, not the ceiling. The /loop agent is encouraged to innovate on algorithms, envs, training recipes, observation schemes, research ideas — and is required to follow these rituals while doing so. Speed without rigor isn't progress.
- **S-11 — Spec is authoritative; spec evolves.** If a rule conflicts with discovered reality (e.g., an invariant turns out to be impractical on the hybrid Gated-DeltaNet + MoE architecture), amend the spec via `docs/superpowers/specs/amendments/` rather than silently skipping. Git history is the audit trail.
- **S-12 — Every PR is a contribution.** Even a tiny change updates `CHANGELOG.md` with { what, why, evidence, invariants-run }. Makes the loop's trail legible to outside contributors and future sessions.

---

## §12. Documentation deliverable

Docs are **always-on** — written incrementally per S-1..S-12, so every new file lands with its doc entry. On top of that there is a **first-milestone comprehensive pass** and a **recurring documentation audit** the /loop agent runs periodically.

### Always-on docs (live alongside code; updated per rituals)

- `README.md` — 1 page: pitch (1 paragraph) + quickstart (5 lines) + link to docs site + link to `docs/RESULTS.md` + citation snippet.
- `docs/index.md` — mkdocs landing page.
- `docs/ARCHITECTURE.md` — library layout, rollout/train split, adapter-sync pattern, MoE + Gated-DeltaNet specifics, where each correctness invariant lives.
- `docs/ALGORITHMS.md` — per algorithm: 1-page math + code pointer + objective + known failure modes.
- `docs/ENVS.md` — table: { env_id, horizon, obs shape, action space, target score, prompt sample, link to probe report }.
- `docs/BACKBONES.md` — table: { name, params, context, footprint at batch N, notes on DeltaNet/MoE/thinking-mode }.
- `docs/RECIPES.md` — copy-pasteable commands to reproduce every Tier-2 green curve + expected wall-clock.
- `docs/RESULTS.md` — live dashboard (auto-generated from metric CSVs + configs).
- `docs/RESEARCH.md` — research journal (auto-appended on every experimental outcome).
- `docs/INVARIANTS.md` — Inv-1..Inv-15 each with { what it checks, how it fails, example bug it caught, link to test file }.
- `docs/CHECKPOINTING.md` — §10 summary + resume walkthroughs + common pitfalls.
- `docs/LOGGING.md` — wandb/rich/CSV interplay + metric glossary.
- `docs/CONTRIBUTING.md` — how to add env/algo/backbone following §11 rituals.
- `docs/TROUBLESHOOTING.md` — OOM, NaN, logprob drift, vLLM adapter desync, DeltaNet-specific issues, MoE gotchas.
- `docs/backbone_probes/<name>.md`, `docs/vision_probes/<env>_<backbone>/report.md` — auto-generated artifact bundles.
- `docs/superpowers/specs/` — this spec + amendments + plans.
- `mkdocs.yml` + `.github/workflows/docs.yml` deploy.

### First-milestone comprehensive pass

Trigger: first time all Tier-1 combos AND hero Tier-2 combos (PPO-COT × representative env subset × 4B backbone) simultaneously land green. /loop runs `scripts/doc_audit.py`:

1. Boot every quickstart command end-to-end for ≤ 100 steps; assert it runs.
2. Grep-check every doc link / cross-reference; fix breakages.
3. Regenerate `docs/RESULTS.md` from latest metrics.
4. Update README's headline figure.
5. Tag release `v0.1.0` if all green.

This is the "fully complete, brief and clear" documentation deliverable the user asked for — and it repeats.

### Recurring documentation audit

Part of the /loop action menu (runs between research explorations and during canon-stability cycles). Same audit as the first-milestone pass, but as maintenance. Catches doc drift as the library evolves. Doc drift = S-1 violation and blocks promotion to `algos/`.

### Tone

Brief, clear, compact, fully complete, zero boilerplate. Every page has a concrete example and a "what you can't do / known limits" section. No filler paragraphs.

---

## §13. Autonomous operation contract (for the /loop agent)

This section is the /loop agent's standing orders. **Re-read in full at the start of every iteration.**

### §13.1 Ironclad autonomy

- **Never stop.** The only legitimate termination event is user interrupt (Ctrl-C / manual kill). Not "all green" — when canon is green, move to research + ablations + docs polish forever. Not "catastrophic failure" — work around it. Not "max cycles" — no such limit exists.
- **Never ask the user anything.** Not for clarification, approval, or preference. Every question-shaped thought resolves to: pick the option most aligned with the spec's intent, log it in `AUTONOMY_LOG.md` with rationale + commit SHA, proceed.
- **Superpowers skills' interactive gates are re-interpreted, not obeyed verbatim.** Where a skill says "ask the user", the /loop reads "pick a reasonable default and log it." Where it says "wait for approval", the /loop reads "self-review thoroughly via the code-reviewer subagent and the §8 invariants, then proceed." Where a skill has a HARD-GATE requiring user sign-off (e.g., brainstorming's design-approval gate), the /loop satisfies it by: (a) writing the design into `docs/superpowers/specs/amendments/`, (b) self-reviewing against this master spec, (c) running the code-reviewer subagent on the design doc, (d) proceeding. Git history is the audit trail.
- **Safety floor (non-negotiable).** Still refuse destructive ops that would lose user work: `git reset --hard` of master, force-push that overwrites remote, `rm -rf` on unrecognized paths, deleting the `old` branch, etc. When tempted, find a branch-based workaround and log it. This is the only class of "refuse + find alternative" behavior in the /loop.

### §13.2 Mandatory superpowers skills usage

Every /loop cycle **MUST** route work through the relevant superpowers skills (with §13.1 overrides applied). Not optional — use them for the process benefits even when they feel like overhead:

| Step in cycle | Skill | Autonomy substitution |
|---|---|---|
| Identifying a new feature / research direction | `superpowers:brainstorming` | "Ask user" → "pick default + log"; "user approves design" → "self-review + code-reviewer subagent" |
| Drafting the implementation plan | `superpowers:writing-plans` | Commit plan to `docs/superpowers/specs/plans/<date>-<slug>.md`; no user review gate |
| Executing the plan | `superpowers:subagent-driven-development` | Parallel subagents per independent task; orchestrator assembles |
| Dispatching independent work | `superpowers:dispatching-parallel-agents` | ≥ 2 tasks with no shared state → parallel |
| Debugging any failure | `superpowers:systematic-debugging` | Used before every fix, not only hard ones |
| Writing new code with tests | `superpowers:test-driven-development` | Red → green → refactor, strict |
| Verifying work is done | `superpowers:verification-before-completion` | Evidence (§8 invariants + targeted tests + evidence gates) before any "done" claim |
| Reviewing finished work | `superpowers:code-reviewer` subagent (via Agent tool) | Independent review; fixes before merging |
| Cleaning up | `simplify` skill | Always after finishing a feature, before committing |
| Isolating risky work | `superpowers:using-git-worktrees` | For research / experimental branches |
| Finishing a branch | `superpowers:finishing-a-development-branch` | Decide merge strategy without asking |

### §13.3 Canonical /loop iteration shape

1. **Orient.** Read master spec (this file) + `LOOP_STATE.md` + last 20 entries of `AUTONOMY_LOG.md` + `CHANGELOG.md` tail.
2. **Pick next task.** Priority order (loose; agent judges): (a) fix a red canon combo; (b) fix a failing invariant; (c) resume a paused Tier-2 run; (d) promote / park an experimental method; (e) onboard a new env / backbone / algo per §11 rituals; (f) run a new research idea; (g) docs audit; (h) polish / simplify.
3. **Brainstorm (adapted).** Use `superpowers:brainstorming` to design the chosen task; self-approve via §13.1 substitution; write to `docs/superpowers/specs/amendments/` if it extends this spec.
4. **Plan.** Use `superpowers:writing-plans` → commit plan file.
5. **Execute.** Use `superpowers:subagent-driven-development`. Parallelize wherever independent.
6. **Verify.** Use `superpowers:verification-before-completion`. Run the §8 invariants that apply. Confirm evidence gates (§9). Evidence precedes any "done" claim.
7. **Review.** Launch `superpowers:code-reviewer` subagent on the diff. Fix findings before merge.
8. **Simplify.** Run `simplify` skill on the changed files.
9. **Commit + update journals.** `CHANGELOG.md` append; `AUTONOMY_LOG.md` append with all decisions; `LOOP_STATE.md` update (done-task + next-task).
10. **Dashboard refresh.** If Tier-2 results changed, regenerate `docs/RESULTS.md`.
11. **Schedule next.** Self-pace via the available scheduling mechanism (no interval specified → agent decides cadence).

### §13.4 Anti-patterns ruled out

| Thought | Correct action |
|---|---|
| "Let me ask the user to clarify X" | NO — pick the spec-aligned default, log rationale, proceed |
| "I'll wait for approval before merging" | NO — self-approve via code-reviewer subagent + invariants + tests |
| "This is ambiguous; better check in" | NO — pick the interpretation closest to spec intent, log it |
| "Let me just quickly fix it without a test" | NO — S-1 applies; tests land in the same commit |
| "I'll skip the invariant this time, it's obvious" | NO — §8's point is that silent bugs pass obvious-looking code |
| "This optimization doesn't need parity because it's 'equivalent'" | NO — S-3; prove Inv-4 parity before merging |
| "I'll re-implement vLLM LoRA hotswap from memory" | NO — S-5; web-fetch current vLLM docs first, cite in PR |
| "I should stop when canon goes green" | NO — research + ablations + docs audits continue forever |
| "Subagent returned an error; I'll skip that task" | NO — debug via systematic-debugging and try again |
| "Design is obvious; I'll skip brainstorming" | NO — §13.2 mandates the skill even for obvious work |
| "This numerical threshold says red, so red" | NO — §0; investigate, understand, iterate; only genuine-correctness failures are hard red |

### §13.5 Thresholds are signals (§0 reinforced here)

When a threshold fires inside a /loop iteration, the response is always **investigate → understand → iterate**. Hard failures are reserved for genuine correctness bugs (shape mismatches, NaNs, true training↔inference divergence, checkpoint round-trip breakage, non-determinism under fixed seed).

### §13.6 Kickoff prompt (what the user types)

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

### §13.7 CLAUDE.md at repo root

The scaffold PR rewrites `CLAUDE.md` to carry §13 verbatim at its top. Because CLAUDE.md sits at the top of the instruction priority order (above superpowers skills), this guarantees every session of the /loop agent reads "never ask, never stop" at session start and knows to apply the §13.1 substitutions when a skill HARD-GATE fires.

---

## §14. Design defaults (open to spec-extend)

Short list of choices the agent takes as defaults; each open to change via spec-extend with evidence.

- **Hyperparameter defaults per env** — seeded from the existing repo's atari PPO config + CleanRL's `ppo_atari.py` + vizdoom config from the current `src/utils/args.py`. Frozen at scaffold; agent updates via spec-extend when better numbers are established.
- **Frame stack** — 4 horizontally-tiled frames on atari/vizdoom; single frame on minigrid.
- **Reward scaling** — `0.01` (matches current code); ablated raw vs scaled vs clipped as part of Q11's ablations.
- **`max_new_tokens` for COT** — default 512 (balance between reasoning budget and vLLM throughput). Ablated {128, 512, 1024}.
- **LoRA rank** — default 32, alpha 64. Ablated {8, 32, 128}.
- **Thinking mode** — on for 4B (native default), off for 0.8B (loop-prone); agent can opt 0.8B in via a loop-detection guardrail.
- **Seeds per combo** — 3 for Tier-2 "green" judgments; 1 for Tier-1 smoke.
- **vLLM version** — whichever current as of first implementation, with S-5 fetch.
- **Backbone onboarding first run** — always runs Inv-1..Inv-15 + §4 probes + `docs/backbone_probes/<name>.md` auto-write.
- **Paper-targeted ablations (at v1)** — LoRA rank, frozen-vision (LoRA on vision tower vs not), reasoning budget (`max_new_tokens`), FP16 vs BF16, action-interface comparison (COT vs action vs head).

---

## Glossary

- **canon** — the 9 curated trainers under `algos/` + 3 baselines under `baselines/`. Changes go through §11 rituals; promotion only after evidence.
- **experimental** — playground under `experimental/` for novel research methods; promoted to `algos/` per §6.
- **hero interface** — COT. All Tier-2 paper runs.
- **hero paper claim** — VLM + LoRA + correct RL → outperforms from-scratch CNN in interactive visual envs, at a small parameter-training budget.
- **Tier-1** — CI-fast smoke. 0.8B backbone, 3 envs, ≤ 10 min wall-clock. Gates PRs.
- **Tier-2** — overnight / manual paper runs. 4B backbone, all envs.
- **green** — combo passes §9 gates by the agent's judgment (§0).
- **red** — combo has a genuine-correctness failure.
- **yellow** — in-progress / investigating.
- **/loop** — the autonomous iteration mechanism running this spec. See §13.
- **S-*** — standing rule, §11.
- **Inv-*** — correctness invariant, §8.

---

## Appendix A: Kickoff prompt

See §13.6 above. The user types the block verbatim to start the autonomous session.

## Appendix B: CLAUDE.md rewrite (at scaffold time)

The scaffold PR replaces the existing `CLAUDE.md` with a new file whose top carries §13 (the autonomous operation contract) verbatim, followed by the usual brief codebase orientation (commands, architecture pointers, important paths). Because CLAUDE.md sits above superpowers skills in the instruction-priority stack, this ensures the /loop's "never ask, never stop" contract is always in effect.

## Appendix C: External references consulted during spec authorship

- Qwen3.5-VL-4B model card (`https://huggingface.co/Qwen/Qwen3.5-4B`) — hybrid Gated-DeltaNet + Gated-Attention + MoE architecture; native `<think>` thinking mode; 262K context; MTP / speculative decoding; Apache-2.0.
- Qwen3.5-VL-0.8B model card (`https://huggingface.co/Qwen/Qwen3.5-0.8B`) — same family, smaller; thinking off by default (loop-prone).
- CleanRL repo (`cleanrl-master` reference zip supplied by user) — `pyproject.toml` with optional-dependencies; `cleanrl_utils/`; `tests/` per-trainer; `benchmark/`; MIT; mkdocs; `uv.lock` — mirrored for `cleanrl-vlm` layout.
- Existing `clean-llm-rl` repo (current `master`, to be moved to branch `old`) — source of three existing trainer variants (COT, action-scoring, MLP-head), current dual-adapter LoRA design, vizdoom action wrappers, per-env prompt conventions.

## Appendix D: Open questions deferred to spec-extend

These are explicitly **not** blockers for the /loop agent to start. Each is a research / engineering question the agent can extend the spec on when evidence accrues:

1. Whether PEFT cleanly handles per-expert LoRA on the sparse-MoE layers; if not, fallback is LoRA on the router only.
2. Whether FSDP2 matches DS-ZeRO-2 on Gated-DeltaNet + MoE hybrid architecture in practice; default is DS-ZeRO-2 until agent establishes otherwise.
3. Whether thinking-mode COT (`<think>`) outperforms engineered `THINK:` / `ACTION:` prompt scaffolding on the paper envs; agent ablates.
4. Whether the pragmatic sub-trajectory GRPO default holds up vs. more principled long-horizon credit assignment methods developed in `experimental/`.
5. Whether adding KL-to-base back into the loss (re-introducing a ref model) improves long-horizon stability despite the simplicity cost.
