# Environments

## Onboarded

| env_id | horizon | obs shape | actions | target | prompt | probe |
|---|---|---|---|---|---|---|
| VizdoomBasic-v1 | 300 tics / ~75 env steps @ frame_skip=4 | (240, 320, 3) uint8 | Discrete(3): MOVE_LEFT / MOVE_RIGHT / ATTACK | 60.0 | `src/cleanrl_vlm/prompts/templates/vizdoom/basic/actor.txt` | [report](vision_probes/VizdoomBasic-v1_qwen3-vl-2b-instruct/report.md) |
| ALE/Pong-v5 | 27000 env steps (master-spec §4 fixed) | (210, 160, 3) uint8 | Discrete(6): NOOP / FIRE / RIGHT / LEFT / RIGHTFIRE / LEFTFIRE | 21.0 | `src/cleanrl_vlm/prompts/templates/atari/pong/actor.txt` | artifact TBD |

Placeholder — remaining envs populated as onboarded per master-spec §11 S-7.

## Planned tiering

- **Tier-1** (CI smoke, 2B backbone, ≤ 10 min): `VizdoomBasic-v1`, `ALE/Pong-v5`, `MiniGrid-Empty-5x5-v0`.
- **Tier-2** (overnight / paper runs, 4B backbone): all remaining ViZDoom scenarios (`VizdoomDeadlyCorridor-v1`, `VizdoomDefendLine-v1`, …), full ALE suite, full Minigrid/BabyAI suite.

Note: `VizdoomBasic-v0` / `VizdoomCorridor-v0` names are deprecated in the current `vizdoom.gymnasium_wrapper`; use the v1 series (see 2026-04-20 backbone-names amendment for rationale).

## Atari horizon

`max_episode_steps = 27000` is **fixed**. See master-spec §4 + §5.

## Target-score table

Lives in `configs/targets.yaml` once populated by /loop.
