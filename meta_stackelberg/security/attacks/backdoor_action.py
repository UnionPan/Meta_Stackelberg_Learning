"""Three-dimensional paper BRL backdoor action and strict codec."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class BackdoorAction:
    """Poison fraction, malicious learning rate, and malicious local epochs."""

    poison_fraction: float
    learning_rate: float
    local_epochs: int

    def __post_init__(self) -> None:
        poison_fraction = _bounded_real(
            self.poison_fraction, 'poison_fraction', low=0.0, high=1.0,
        )
        learning_rate = _bounded_real(
            self.learning_rate, 'learning_rate', low=0.0, high=0.1,
        )
        if (
            isinstance(self.local_epochs, bool)
            or not isinstance(self.local_epochs, Integral)
            or not 1 <= int(self.local_epochs) <= 10
        ):
            raise ValueError('local_epochs must be an integer within [1, 10]')
        object.__setattr__(self, 'poison_fraction', poison_fraction)
        object.__setattr__(self, 'learning_rate', learning_rate)
        object.__setattr__(self, 'local_epochs', int(self.local_epochs))

    def as_tuple(self) -> tuple[float, float, int]:
        return self.poison_fraction, self.learning_rate, self.local_epochs


@dataclass(frozen=True)
class BackdoorActionCodec:
    """Decode raw TD3 actions using the public RLBackdoorFL 3-D contract."""

    def decode(self, raw_action: np.ndarray) -> BackdoorAction:
        raw = _raw3(raw_action)
        poison_index = min(10, max(0, int(raw[0] * 5.5 + 5.5)))
        learning_rate = (raw[1] + 1.0) * 0.05
        local_epochs = min(10, max(1, int(raw[2] * 5.0 + 6.0)))
        return BackdoorAction(
            poison_fraction=poison_index / 10.0,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
        )


def _raw3(value: np.ndarray) -> tuple[float, float, float]:
    candidate = np.asarray(value)
    if candidate.shape != (3,):
        raise ValueError('raw action must have shape (3,)')
    if not np.issubdtype(candidate.dtype, np.floating):
        raise TypeError('raw action must have floating dtype')
    result = tuple(float(item) for item in candidate)
    if any(not math.isfinite(item) for item in result):
        raise ValueError('raw action must be finite')
    if any(item < -1.0 or item > 1.0 for item in result):
        raise ValueError('raw action must be within [-1, 1]')
    return result  # type: ignore[return-value]


def _bounded_real(value: float, name: str, *, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if result < low or result > high:
        raise ValueError(f'{name} must be within [{low}, {high}]')
    return result
