"""Trainer protocol and factory for Tianshou-backed RL attackers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig


@dataclass
class CollectStats:
    steps: int
    reward_mean: float
    info_means: dict[str, float] = field(default_factory=dict)


@dataclass
class UpdateStats:
    gradient_steps: int
    loss: float


class Trainer(Protocol):
    def ensure_initialized(self, obs_space, action_space) -> None: ...
    def warmup_collect(self, env, random_steps: int, *, reset_on_start: bool = True) -> CollectStats: ...
    def collect(self, env, steps: int, *, reset_on_start: bool = True) -> CollectStats: ...
    def add_transition(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        *,
        reward: float,
        obs_next: np.ndarray,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None: ...
    def update(self, gradient_steps: int) -> UpdateStats: ...
    def act(self, obs, *, deterministic: bool = False) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def diagnostics(self) -> dict[str, float]: ...


def build_trainer(config: RLAttackerConfig) -> Trainer:
    algorithm = config.algorithm.lower()
    try:
        if algorithm == "td3":
            from fl_sandbox.attacks.rl_attacker.tianshou_backend.td3 import TianshouTD3Trainer

            return TianshouTD3Trainer(config)
    except ImportError as exc:
        raise RuntimeError(
            "Tianshou and Gymnasium are required for the RL attacker trainer. "
            "Install with: python -m pip install tianshou gymnasium"
        ) from exc
    raise ValueError(f"Paper RL attacker requires TD3, got: {config.algorithm}")
