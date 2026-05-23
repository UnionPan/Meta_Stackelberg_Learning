"""Reward helpers for RL backdoor attacker training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackdoorRewardInputs:
    clean_loss: float = 0.0
    backdoor_loss: float = 0.0
    sampled_attacker_count: int = 1
    asr_before: float = 0.0
    asr_after: float = 0.0
    clean_before: float = 0.0
    clean_after: float = 0.0
    norm_ratio: float = 1.0


class BackdoorRewardFn:
    def __init__(
        self,
        *,
        mode: str = "paper",
        clean_lambda: float = 0.5,
        clean_weight: float = 1.0,
        norm_weight: float = 0.1,
    ) -> None:
        self.mode = str(mode).lower()
        self.clean_lambda = float(clean_lambda)
        self.clean_weight = float(clean_weight)
        self.norm_weight = float(norm_weight)

    def __call__(self, inputs: BackdoorRewardInputs) -> float:
        mode = self.mode
        if mode in {"paper", "henger_li", "henger-li"}:
            if int(inputs.sampled_attacker_count) <= 0:
                return 0.0
            clean_lambda = min(1.0, max(0.0, self.clean_lambda))
            return -(
                clean_lambda * float(inputs.clean_loss)
                + (1.0 - clean_lambda) * float(inputs.backdoor_loss)
            )

        asr_gain = float(inputs.asr_after) - float(inputs.asr_before)
        if mode in {"delta", "asr_delta", "poi_delta"}:
            return asr_gain

        clean_drop = max(0.0, float(inputs.clean_before) - float(inputs.clean_after))
        norm_penalty = max(0.0, float(inputs.norm_ratio) - 1.0)
        return asr_gain - self.clean_weight * clean_drop - self.norm_weight * norm_penalty
