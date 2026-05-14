"""MiniGrid env factories (gymnasium-registered via ``minigrid``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from cleanrl_vlm.envs.wrappers import DictImageKeyWrapper


def make_minigrid_env(env_id: str, config: dict[str, Any]) -> Callable[[], gym.Env]:
    """Return a thunk that builds a MiniGrid env with RGB-image obs.

    Wrapping order:
        gym.make -> RGBImg{,Partial}ObsWrapper(tile_size=...) ->
        DictImageKeyWrapper -> RecordEpisodeStatistics

    The RGB wrapper replaces ``obs["image"]`` (which is symbolic by
    default) with a rendered RGB frame. ``DictImageKeyWrapper`` then
    drops the other Dict entries (``direction``, ``mission``) so
    downstream code sees a plain ``[H, W, 3]`` uint8 image, matching
    VizDoom + Atari.

    Observation mode (``obs_mode`` in per-env YAML):
      - ``"full"`` *(default)*: ``RGBImgObsWrapper`` — the full grid is
        visible to the VLM. Best when the grid is small enough to fit
        in the VLM's context and the agent's position / orientation is
        visible in the render. Preferred for 5x5–8x8 Empty envs.
      - ``"partial"``: ``RGBImgPartialObsWrapper`` — classic agent-
        centered 7x7 view. Used for the standard partial-observability
        ablation; keeps the task aligned with the CNN-PPO baseline
        literature.

    Single-frame by default: MiniGrid's state is already symbolic-rich,
    so frame-stacking usually hurts more than helps.
    """
    import minigrid  # noqa: F401  — registers envs with gymnasium
    from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper

    tile_size = int(config.get("tile_size", 32))
    max_episode_steps = config.get("max_episode_steps")
    obs_mode = str(config.get("obs_mode", "full")).lower()
    if obs_mode == "full":
        rgb_wrapper: type[gym.ObservationWrapper] = RGBImgObsWrapper
    elif obs_mode == "partial":
        rgb_wrapper = RGBImgPartialObsWrapper
    else:
        raise ValueError(f"Unknown obs_mode={obs_mode!r}; expected 'full' or 'partial'.")

    def thunk() -> gym.Env:
        make_kwargs: dict[str, Any] = {}
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = int(max_episode_steps)
        env = gym.make(env_id, **make_kwargs)
        env = rgb_wrapper(env, tile_size=tile_size)
        env = DictImageKeyWrapper(env, key="image")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
