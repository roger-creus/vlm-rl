"""Gymnasium wrappers for cleanrl-vlm env layer."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameSkipEnv(gym.Wrapper):
    """Repeat the same action `skip` times, sum reward, return final obs."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """Drop ViZDoom dict obs down to just the `screen` RGB array."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "ScreenOnlyWrapper expects Dict obs"
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return np.asarray(obs["screen"])


class DiscreteMultiBinaryWrapper(gym.ActionWrapper):
    """Generalizes DeadlyCorridor / DefendTheLine prototype wrappers.

    Converts a MultiBinary(N)-style button space into Discrete(N) by setting
    exactly one button to 1 per step. Button names are recorded for logging
    and prompt construction.
    """

    def __init__(self, env: gym.Env, button_names: list[str]) -> None:
        super().__init__(env)
        self.button_names = list(button_names)
        self.action_space = spaces.Discrete(len(self.button_names))

    def action(self, act):
        arr = [0] * len(self.button_names)
        if 0 <= int(act) < len(self.button_names):
            arr[int(act)] = 1
        return arr
