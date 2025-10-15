import torch
import gymnasium as gym
from vizdoom import gymnasium_wrapper 
from IPython import embed
import matplotlib.pyplot as plt

from src.utils.utils import make_vizdoom_env

envs = gym.vector.SyncVectorEnv(
    [make_vizdoom_env("VizdoomCorridor-v0", i, False, "test") for i in range(8)],
)

done = torch.zeros(8)
obs, _ = envs.reset()
print(obs.shape)
while not done.any():
    action = envs.action_space.sample()
    print(action)
    obs, reward, done, trunc, info = envs.step(action)
    print(info.keys())
    