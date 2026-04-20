# Recipes

Copy-pasteable commands to reproduce every Tier-2 green curve. Populated by `/loop` as curves land.

## Bootstrap smoke

```bash
uv sync --extra dev
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv run pytest -m tier1 -v
```

## PPO-COT on VizdoomBasic with Qwen3-VL-2B-Instruct

```bash
source scripts/_cluster_env.sh
uv run --no-env-file python -u -m algos.ppo_cot \
    --env-id VizdoomBasic-v1 \
    --backbone Qwen/Qwen3-VL-2B-Instruct \
    --num-envs 4 --num-steps 32 \
    --total-timesteps 200000 \
    --max-new-tokens 256 --seed 0 --track
```

Expected wall-clock on a single A6000: ~2.5-3 min per iteration at
`num_envs=4 num_steps=32 max_new_tokens=256`; a 100-iter run ≈ 4-5 h.
Per spec §0 the target is "genuinely learning" by agent judgment rather
than a hard score threshold; `configs/targets.yaml` seeds a reference
target of 60.0 on `VizdoomBasic-v1`.

**Small smoke variant** (for iteration-on-code; ~15 min wall):

```bash
uv run --no-env-file python -u -m algos.ppo_cot \
    --env-id VizdoomBasic-v1 --num-envs 2 --num-steps 16 \
    --total-timesteps 320 --max-new-tokens 64 \
    --num-minibatches 2 --update-epochs 2 --checkpoint-interval 5
```
