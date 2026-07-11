"""Immutable physical actions used by aggregation defenses."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DefenseAction:
    clip_radius: float
    trim_ratio: float = 0.0

    def __post_init__(self) -> None:
        radius = _finite_real(self.clip_radius, 'clip_radius')
        if radius <= 0.0:
            raise ValueError('clip_radius must be finite and positive')
        ratio = _finite_real(self.trim_ratio, 'trim_ratio')
        if ratio < 0.0 or ratio >= 0.5:
            raise ValueError('trim_ratio must be within [0, 0.5)')
        object.__setattr__(self, 'clip_radius', radius)
        object.__setattr__(self, 'trim_ratio', ratio)


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a real number') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result
