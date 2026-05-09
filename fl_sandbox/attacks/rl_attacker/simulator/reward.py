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


class RunningNormalizer:
    """Online scalar normalizer with stable early-step behavior."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def normalize(self, value: float) -> float:
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.count < 2:
            return 0.0
        variance = max(self.m2 / max(1, self.count - 1), 1e-8)
        return float((value - self.mean) / np.sqrt(variance))


class DefaultRewardFn:
    def __init__(self, config: RLAttackerConfig) -> None:
        self.config = config
        self.normalizers = {
            "loss": RunningNormalizer(),
            "acc": RunningNormalizer(),
            "bypass": RunningNormalizer(),
            "norm": RunningNormalizer(),
        }
        self.last_components: dict[str, float] = {}

    def __call__(self, inputs: RewardInputs) -> float:
        smoothness = float(np.sum((inputs.action - inputs.previous_action) ** 2))
        saturation = float(np.mean(np.isclose(np.abs(inputs.action), 1.0)))
        components = {
            "loss": self.normalizers["loss"].normalize(inputs.loss_delta),
            "acc": self.normalizers["acc"].normalize(inputs.acc_delta),
            "bypass": self.normalizers["bypass"].normalize(inputs.bypass_score),
            "norm": self.normalizers["norm"].normalize(inputs.norm_penalty),
            "smoothness": smoothness,
            "oob": saturation,
        }
        self.last_components = components
        return float(
            self.config.reward_loss_weight * components["loss"]
            + self.config.reward_accuracy_weight * components["acc"]
            + self.config.reward_bypass_weight * components["bypass"]
            - self.config.reward_norm_penalty_weight * components["norm"]
            - self.config.reward_action_smoothness_weight * smoothness
            - self.config.reward_action_saturation_weight * saturation
        )
