"""ViZDoom env factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from cleanrl_vlm.envs.vizdoom.action_tables import action_tables
from cleanrl_vlm.envs.wrappers import (
    DiscreteMultiBinaryWrapper,
    ScreenOnlyWrapper,
)


def make_vizdoom_env(env_id: str, config: dict[str, Any]) -> Callable[[], gym.Env]:
    """Return a thunk that builds the wrapped env.

    Wrapper order: gym.make (with frame_skip kwarg baked in) ->
    DiscreteMultiBinaryWrapper -> ScreenOnlyWrapper -> RecordEpisodeStatistics.
    """
    # Ensure the vizdoom gymnasium wrapper is registered.
    from vizdoom import gymnasium_wrapper  # noqa: F401

    if env_id not in action_tables:
        raise KeyError(f"Unknown ViZDoom env {env_id!r}; extend action_tables.")
    buttons = action_tables[env_id]
    frame_skip = int(config.get("frame_skip", 4))

    def thunk() -> gym.Env:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            max_buttons_pressed=0,
            frame_skip=frame_skip,
        )
        env = DiscreteMultiBinaryWrapper(env, button_names=buttons)
        env = ScreenOnlyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
