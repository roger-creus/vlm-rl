# vlm-rl

Research code for online reinforcement learning with vision-language models in visual environments.

The current experiment is PPO-COT: a VLM sees an RGB observation, generates a short response ending in `ACTION: <NAME>`, and PPO updates LoRA adapters plus a small critic head from on-policy rollouts.

## Status

This is an active research repo, not a finished benchmark package.

Implemented:

- PPO-COT trainer in `algos/ppo_cot.py`
- Qwen3.5-VL model wrapper with actor/critic LoRA adapters
- ViZDoom Basic, ALE Pong, and MiniGrid Empty factories
- Prompt templates under `src/cleanrl_vlm/prompts/templates/`
- Unit and invariant tests for parsing, rollout storage, GAE, LoRA wiring, logging, checkpoints, reward flow, determinism, and padding masks

Not implemented yet:

- vLLM rollout serving
- GRPO/RLOO variants
- CNN and frozen-VLM baselines
- long multi-seed result tables
- full checkpoint/resume parity

## Install

Use Python 3.10 and `uv`.

```bash
git clone git@github.com:roger-creus/vlm-rl.git
cd vlm-rl

uv venv --python 3.10
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra dev --extra distributed   # accelerate/deepspeed/vllm
uv sync --extra dev --extra tracking      # wandb/tensorboard
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`transformers` is installed from GitHub because the active Qwen3.5-VL path depends on recent upstream support.

## Check

CPU-safe checks:

```bash
uv run --no-env-file ruff check .
uv run --no-env-file ruff format --check .
uv run --no-env-file pyright src/cleanrl_vlm
uv run --no-env-file pytest tests/test_imports.py tests/unit tests/invariants -v
```

GPU smoke:

```bash
uv run --no-env-file pytest tests/smoke -m "tier1 and gpu" -v
```

## Run

Small ViZDoom smoke run:

```bash
uv run --no-env-file python -m algos.ppo_cot \
  --env-id VizdoomBasic-v1 \
  --env-config configs/envs/VizdoomBasic-v1.yaml \
  --backbone Qwen/Qwen3.5-2B \
  --num-envs 2 \
  --num-steps 16 \
  --total-timesteps 320 \
  --max-new-tokens 64 \
  --num-minibatches 2 \
  --update-epochs 2
```

Run outputs go to `runs/`, which is gitignored.

## Layout

- `algos/ppo_cot.py` — main trainer
- `src/cleanrl_vlm/` — package code
- `configs/` — env/backbone/target config
- `scripts/` — local probe scripts
- `tests/` — test suite

## License

MIT. See `LICENSE`.
