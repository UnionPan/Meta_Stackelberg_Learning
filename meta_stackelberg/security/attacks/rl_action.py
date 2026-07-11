"""Three-dimensional model-based RL attack action and strict codec."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from meta_stackelberg.security.defenses.paper_action import _encode, _finite, _raw3


@dataclass(frozen=True)
class RLAttackAction:
    gamma: float
    local_steps: int
    stealth_lambda: float

    def __post_init__(self) -> None:
        gamma = _finite(self.gamma, 'gamma')
        if gamma < 0.0:
            raise ValueError('gamma must be non-negative')
        if isinstance(self.local_steps, bool) or not isinstance(self.local_steps, int):
            raise TypeError('local_steps must be an integer')
        if self.local_steps <= 0:
            raise ValueError('local_steps must be positive')
        stealth = _finite(self.stealth_lambda, 'stealth_lambda')
        if stealth < 0.0 or stealth > 1.0:
            raise ValueError('stealth_lambda must be within [0, 1]')
        object.__setattr__(self, 'gamma', gamma)
        object.__setattr__(self, 'stealth_lambda', stealth)


@dataclass(frozen=True)
class RLAttackActionCodec:
    gamma_min: float = 0.1
    gamma_max: float = 2.9
    local_steps_min: int = 1
    local_steps_max: int = 19
    stealth_min: float = 0.05
    stealth_max: float = 0.95

    def decode(self, raw_action: np.ndarray) -> RLAttackAction:
        raw = _raw3(raw_action)
        continuous_steps = self.local_steps_min + (raw[1] + 1.0) * 0.5 * (
            self.local_steps_max - self.local_steps_min
        )
        steps = int(math.floor(continuous_steps + 0.5))
        return RLAttackAction(
            _canonical(self.gamma_min + (raw[0] + 1.0) * 0.5 * (
                self.gamma_max - self.gamma_min
            )),
            steps,
            _canonical(self.stealth_min + (raw[2] + 1.0) * 0.5 * (
                self.stealth_max - self.stealth_min
            )),
        )

    def encode(self, action: RLAttackAction) -> np.ndarray:
        if not (
            self.gamma_min <= action.gamma <= self.gamma_max
            and self.local_steps_min <= action.local_steps <= self.local_steps_max
            and self.stealth_min <= action.stealth_lambda <= self.stealth_max
        ):
            raise ValueError('RL attack action is outside codec bounds')
        return np.array([
            _encode(action.gamma, self.gamma_min, self.gamma_max),
            _encode(action.local_steps, self.local_steps_min, self.local_steps_max),
            _encode(action.stealth_lambda, self.stealth_min, self.stealth_max),
        ], dtype=np.float64)


def _canonical(value: float) -> float:
    return float(round(value, 15))
