"""Three-dimensional paper Defender action and strict codec."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PaperDefenderAction:
    alpha: float
    beta: float
    epsilon: float

    def __post_init__(self) -> None:
        alpha = _finite(self.alpha, 'alpha')
        beta = _finite(self.beta, 'beta')
        epsilon = _finite(self.epsilon, 'epsilon')
        if alpha <= 0.0:
            raise ValueError('alpha must be positive')
        if beta < 0.0 or beta >= 0.5:
            raise ValueError('beta must be within [0, 0.5)')
        if epsilon <= 0.0:
            raise ValueError('epsilon must be positive')
        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'beta', beta)
        object.__setattr__(self, 'epsilon', epsilon)


@dataclass(frozen=True)
class PaperDefenderActionCodec:
    alpha_min: float = 1e-6
    alpha_floor_ratio: float = 0.0
    beta_max: float = 0.45
    # Appendix D evaluates NeuroClip at epsilon=1, 6, and 10, while
    # Appendix C uses the original epsilon=7 baseline.  The paper does not
    # publish a normalized-action decoder, so keep the declared range inside
    # the values for which the paper provides experimental evidence.
    epsilon_min: float = 1.0
    epsilon_max: float = 10.0

    def __post_init__(self) -> None:
        ratio = _finite(self.alpha_floor_ratio, 'alpha_floor_ratio')
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError('alpha_floor_ratio must be within [0, 1)')
        object.__setattr__(self, 'alpha_floor_ratio', ratio)

    def decode(self, raw_action: np.ndarray, *, observed_max_norm: float) -> PaperDefenderAction:
        raw = _raw3(raw_action)
        maximum = self._maximum(observed_max_norm)
        minimum = self._minimum(maximum)
        return PaperDefenderAction(
            _decode(raw[0], minimum, maximum),
            _decode(raw[1], 0.0, self.beta_max),
            _decode(raw[2], self.epsilon_min, self.epsilon_max),
        )

    def encode(self, action: PaperDefenderAction, *, observed_max_norm: float) -> np.ndarray:
        maximum = self._maximum(observed_max_norm)
        minimum = self._minimum(maximum)
        if not minimum <= action.alpha <= maximum or action.beta > self.beta_max or not (
            self.epsilon_min <= action.epsilon <= self.epsilon_max
        ):
            raise ValueError('paper Defender action is outside codec bounds')
        return np.array([
            _encode(action.alpha, minimum, maximum),
            _encode(action.beta, 0.0, self.beta_max),
            _encode(action.epsilon, self.epsilon_min, self.epsilon_max),
        ], dtype=np.float64)

    def _maximum(self, value: float) -> float:
        maximum = _finite(value, 'observed_max_norm')
        if maximum <= self.alpha_min:
            raise ValueError('observed_max_norm must exceed alpha_min')
        return maximum

    def _minimum(self, maximum: float) -> float:
        return max(self.alpha_min, self.alpha_floor_ratio * maximum)


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


def _decode(value: float, low: float, high: float) -> float:
    return low + (value + 1.0) * 0.5 * (high - low)


def _encode(value: float, low: float, high: float) -> float:
    return 2.0 * (value - low) / (high - low) - 1.0


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result
