"""Reward functions for the RL attacker simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig


@dataclass
class RewardInputs:
    loss_delta: float
    acc_delta: float
    bypass_score: float
    action: np.ndarray
    previous_action: np.ndarray
    norm_penalty: float


class DefaultRewardFn:
    def __init__(self, config: RLAttackerConfig) -> None:
        self.config = config

    def __call__(self, inputs: RewardInputs) -> float:
        smoothness = float(np.sum((inputs.action - inputs.previous_action) ** 2))
        saturation = float(np.mean(np.isclose(np.abs(inputs.action), 1.0)))
        return float(
            inputs.loss_delta
            + self.config.reward_accuracy_weight * inputs.acc_delta
            + self.config.reward_bypass_weight * inputs.bypass_score
            - self.config.reward_norm_penalty_weight * inputs.norm_penalty
            - self.config.reward_action_smoothness_weight * smoothness
            - self.config.reward_action_saturation_weight * saturation
        )
