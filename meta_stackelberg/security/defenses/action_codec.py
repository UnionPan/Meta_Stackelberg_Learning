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


@dataclass(frozen=True)
class ClippedTrimmedActionCodec:
    min_clip_radius: float
    max_clip_radius: float
    max_trim_ratio: float = 0.4

    def __post_init__(self) -> None:
        lower = _positive_finite(self.min_clip_radius, 'min_clip_radius')
        upper = _positive_finite(self.max_clip_radius, 'max_clip_radius')
        if lower >= upper:
            raise ValueError('min_clip_radius must be less than max_clip_radius')
        maximum_trim = _finite_scalar(self.max_trim_ratio, 'max_trim_ratio')
        if maximum_trim <= 0.0 or maximum_trim >= 0.5:
            raise ValueError('max_trim_ratio must be within (0, 0.5)')
        object.__setattr__(self, 'min_clip_radius', lower)
        object.__setattr__(self, 'max_clip_radius', upper)
        object.__setattr__(self, 'max_trim_ratio', maximum_trim)

    def decode(self, normalized_action: np.ndarray) -> DefenseAction:
        clip_raw, trim_raw = _normalized_vector(normalized_action, size=2)
        return DefenseAction(
            _linear_decode(clip_raw, self.min_clip_radius, self.max_clip_radius),
            _linear_decode(trim_raw, 0.0, self.max_trim_ratio),
        )

    def encode(self, action: DefenseAction) -> np.ndarray:
        if (
            action.clip_radius < self.min_clip_radius
            or action.clip_radius > self.max_clip_radius
            or action.trim_ratio > self.max_trim_ratio
        ):
            raise ValueError('defense action is outside codec bounds')
        return np.array([
            _linear_encode(
                action.clip_radius,
                self.min_clip_radius,
                self.max_clip_radius,
            ),
            _linear_encode(action.trim_ratio, 0.0, self.max_trim_ratio),
        ], dtype=np.float64)


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


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a real number') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
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


def _normalized_vector(value: np.ndarray, *, size: int) -> tuple[float, ...]:
    candidate = np.asarray(value)
    if candidate.shape != (size,):
        raise ValueError(f'normalized action must have shape ({size},)')
    if not np.issubdtype(candidate.dtype, np.floating):
        raise TypeError('normalized action must have a floating dtype')
    result = tuple(float(item) for item in candidate)
    if any(not math.isfinite(item) for item in result):
        raise ValueError('normalized action must be finite')
    if any(item < -1.0 or item > 1.0 for item in result):
        raise ValueError('normalized action must be within [-1, 1]')
    return result


def _linear_decode(value: float, lower: float, upper: float) -> float:
    return lower + (value + 1.0) * 0.5 * (upper - lower)


def _linear_encode(value: float, lower: float, upper: float) -> float:
    return 2.0 * (value - lower) / (upper - lower) - 1.0
