"""Single-point env registration / dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym


def make_env(
    env_id: str,
    config: dict[str, Any],
    seed: int,
    idx: int,
    run_name: str,
) -> Callable[[], gym.Env]:
    """Dispatch to the appropriate env sub-package by env-id prefix."""
    if env_id.startswith("Vizdoom"):
        from cleanrl_vlm.envs.vizdoom.factories import make_vizdoom_env

        thunk = make_vizdoom_env(env_id, config)
    else:
        raise KeyError(f"No factory registered for env_id={env_id!r}")

    def seeded_thunk() -> gym.Env:
        env = thunk()
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env

    return seeded_thunk
