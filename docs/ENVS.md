# Environments

Placeholder — populated as envs are onboarded per master-spec §11 S-7.

## Planned tiering

- **Tier-1** (CI smoke, 0.8B backbone, ≤ 10 min): `VizdoomBasic-v0`, `ALE/Pong-v5`, `MiniGrid-Empty-5x5-v0`.
- **Tier-2** (overnight / paper runs, 4B backbone): all remaining ViZDoom scenarios, full ALE suite, full Minigrid/BabyAI suite.

## Atari horizon

`max_episode_steps = 27000` is **fixed**. See master-spec §4 + §5.

## Target-score table

Lives in `configs/targets.yaml` once populated by /loop.
