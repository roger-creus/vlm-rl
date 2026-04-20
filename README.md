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
