# Recipes

Copy-pasteable commands to reproduce every Tier-2 green curve. Populated by `/loop` as curves land.

## Bootstrap smoke

```bash
uv sync --extra dev
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv run pytest -m tier1 -v
```

## First canon trainer (when it lands)

Filled by /loop once `algos/ppo_cot.py` is merged.
