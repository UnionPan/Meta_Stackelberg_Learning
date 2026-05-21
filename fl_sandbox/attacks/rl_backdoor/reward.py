"""Reward helpers for RL backdoor attacker training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackdoorRewardInputs:
    asr_before: float
    asr_after: float
    clean_before: float
    clean_after: float
    norm_ratio: float = 1.0


class BackdoorRewardFn:
    def __init__(self, *, clean_weight: float = 1.0, norm_weight: float = 0.1) -> None:
        self.clean_weight = float(clean_weight)
        self.norm_weight = float(norm_weight)

    def __call__(self, inputs: BackdoorRewardInputs) -> float:
        asr_gain = float(inputs.asr_after) - float(inputs.asr_before)
        clean_drop = max(0.0, float(inputs.clean_before) - float(inputs.clean_after))
        norm_penalty = max(0.0, float(inputs.norm_ratio) - 1.0)
        return asr_gain - self.clean_weight * clean_drop - self.norm_weight * norm_penalty
