"""Atari env factories (ALE-v5 via gymnasium)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym


def make_atari_env(env_id: str, config: dict[str, Any]) -> Callable[[], gym.Env]:
    """Return a thunk that builds an ALE-v5 env with spec §4 defaults.

    - EpisodicLife is NOT applied; ALE full lifetime is the default horizon.
    - Frame-stacking is NOT applied here. VLM-aware frame tiling is planned
      as an explicit wrapper.
    - ALE's native frameskip is used (v5 default = 4, sticky_action_probability=0.25).
    """
    import ale_py

    gym.register_envs(ale_py)

    max_episode_steps = config.get("max_episode_steps", 27_000)
    clip_reward = bool(config.get("clip_reward", False))

    def thunk() -> gym.Env:
        env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda r: float(max(-1.0, min(1.0, float(r)))))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
