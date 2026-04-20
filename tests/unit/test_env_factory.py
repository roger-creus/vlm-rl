import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


def test_frame_skip_repeats_action_and_sums_reward():
    from cleanrl_vlm.envs.wrappers import FrameSkipEnv

    class Dummy(gym.Env):
        observation_space = spaces.Box(0, 255, (4,), dtype=np.uint8)
        action_space = spaces.Discrete(2)

        def __init__(self):
            self.t = 0

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return np.zeros(4, dtype=np.uint8), {}

        def step(self, action):
            self.t += 1
            return np.full(4, self.t, dtype=np.uint8), 1.0, self.t >= 10, False, {}

    env = FrameSkipEnv(Dummy(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert reward == 4.0
    assert obs[0] == 4


def test_screen_only_wrapper_unwraps_dict_obs():
    from cleanrl_vlm.envs.wrappers import ScreenOnlyWrapper

    class DictEnv(gym.Env):
        observation_space = spaces.Dict({"screen": spaces.Box(0, 255, (3, 4, 4), dtype=np.uint8)})
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return {"screen": np.zeros((3, 4, 4), dtype=np.uint8)}, {}

        def step(self, action):
            return {"screen": np.ones((3, 4, 4), dtype=np.uint8)}, 0.0, False, False, {}

    env = ScreenOnlyWrapper(DictEnv())
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 4, 4)


@pytest.mark.parametrize(
    "buttons,chosen_idx",
    [
        (["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"], 2),
        (
            [
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "ATTACK",
                "MOVE_FORWARD",
                "MOVE_BACKWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
            ],
            5,
        ),
        (["TURN_LEFT", "TURN_RIGHT", "ATTACK"], 0),
    ],
)
def test_discrete_multibinary_wrapper_generalizes(buttons, chosen_idx):
    from cleanrl_vlm.envs.wrappers import DiscreteMultiBinaryWrapper

    class MB(gym.Env):
        def __init__(self, n):
            self.action_space = spaces.MultiBinary(n)
            self.observation_space = spaces.Box(0, 255, (1,), dtype=np.uint8)
            self.last_action = None

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.uint8), {}

        def step(self, action):
            self.last_action = list(action)
            return np.zeros(1, dtype=np.uint8), 0.0, False, False, {}

    base = MB(len(buttons))
    env = DiscreteMultiBinaryWrapper(base, button_names=buttons)
    assert env.action_space == spaces.Discrete(len(buttons))
    env.reset()
    env.step(chosen_idx)
    expected = [0] * len(buttons)
    expected[chosen_idx] = 1
    assert base.last_action == expected


def test_action_tables_contains_vizdoom_basic():
    from cleanrl_vlm.envs.vizdoom.action_tables import action_tables

    assert action_tables["VizdoomBasic-v0"] == ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    assert action_tables["VizdoomCorridor-v0"] == [
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
    ]
    assert action_tables["VizdoomDefendLine-v0"] == ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
