"""Rollout buffer + GAE."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def compute_gae(
    rewards: torch.Tensor,  # [T, B]
    values: torch.Tensor,  # [T, B]
    dones: torch.Tensor,  # [T, B]
    next_value: torch.Tensor,  # [B]
    next_done: torch.Tensor,  # [B]
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE(lambda). Returns ``(advantages, returns)``."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


@dataclass
class RolloutBuffer:
    num_envs: int
    num_steps: int
    obs_shape: tuple[int, ...]
    device: torch.device

    obs: torch.Tensor = field(init=False)
    actions: torch.Tensor = field(init=False)
    logprob_sum: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)
    # Variable-length per step: trainer appends one ``[num_envs, S_t]`` LongTensor
    # per rollout step (``S_t`` = prompt_len_t + gen_len_t, padded to max in batch)
    # so the PPO update re-score can pass them back as ``action_ids`` to the
    # actor adapter without re-running ``.generate()``. ``prompt_lens_per_step[t]``
    # is a ``[num_envs]`` LongTensor of per-row prompt lengths.
    full_ids_per_step: list[torch.Tensor | None] = field(init=False)
    prompt_lens_per_step: list[torch.Tensor | None] = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.num_steps, self.num_envs)
        self.obs = torch.zeros((*shape, *self.obs_shape), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.logprob_sum = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.full_ids_per_step = [None] * self.num_steps
        self.prompt_lens_per_step = [None] * self.num_steps

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
    ) -> None:
        adv, ret = compute_gae(
            self.rewards,
            self.values,
            self.dones,
            next_value,
            next_done,
            gamma,
            lam,
        )
        self.advantages = adv
        self.returns = ret
