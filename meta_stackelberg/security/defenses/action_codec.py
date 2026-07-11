"""Mappings between normalized policy output and physical defense actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from meta_stackelberg.security.defenses.actions import DefenseAction


@dataclass(frozen=True)
class ClipRadiusActionCodec:
    min_radius: float
    max_radius: float

    def __post_init__(self) -> None:
        lower = _positive_finite(self.min_radius, 'min_radius')
        upper = _positive_finite(self.max_radius, 'max_radius')
        if lower >= upper:
            raise ValueError('min_radius must be less than max_radius')
        object.__setattr__(self, 'min_radius', lower)
        object.__setattr__(self, 'max_radius', upper)

    def decode(self, normalized_action: np.ndarray) -> DefenseAction:
        value = _normalized_scalar(normalized_action)
        radius = self.min_radius + (value + 1.0) * 0.5 * (
            self.max_radius - self.min_radius
        )
        return DefenseAction(radius)

    def encode(self, action: DefenseAction) -> np.ndarray:
        radius = action.clip_radius
        if radius < self.min_radius or radius > self.max_radius:
            raise ValueError('clip_radius is outside codec bounds')
        value = 2.0 * (radius - self.min_radius) / (
            self.max_radius - self.min_radius
        ) - 1.0
        return np.array([value], dtype=np.float64)


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and positive')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite and positive') from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return result


def _normalized_scalar(value: np.ndarray) -> float:
    candidate = np.asarray(value)
    if candidate.shape != (1,):
        raise ValueError('normalized action must have shape (1,)')
    if not np.issubdtype(candidate.dtype, np.floating):
        raise TypeError('normalized action must have a floating dtype')
    result = float(candidate[0])
    if not math.isfinite(result):
        raise ValueError('normalized action must be finite')
    if result < -1.0 or result > 1.0:
        raise ValueError('normalized action must be within [-1, 1]')
    return result
