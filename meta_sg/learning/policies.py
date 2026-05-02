"""
TD3 Actor and Critic networks.
Both defender and attacker share the same architecture.
Action space: 3-dimensional continuous, squashed to [-1, 1] via tanh.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Deterministic policy π(s; θ) → a ∈ [-1, 1]^act_dim."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """
    Twin Q-networks Q1, Q2 for TD3 double-critic.
    Input: (s, a) concatenated.
    Output: (Q1(s,a), Q2(s,a)).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = obs_dim + act_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


class ConstantActionPolicy:
    """
    Lightweight rollout policy for non-adaptive paper attacks.

    Fixed attacks such as IPM/LMP/BFL/DBA are pre-defined methods in the paper,
    not TD3 attackers. They still receive a compressed 3D action for API
    uniformity, but that action should be fixed unless we intentionally study
    attack-parameter adaptation.
    """

    def __init__(self, action: np.ndarray | None = None, act_dim: int = 3) -> None:
        if action is None:
            action = np.zeros(act_dim, dtype=np.float32)
        self.action = np.asarray(action, dtype=np.float32)

    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.action.copy()
