"""Immutable physical actions used by aggregation defenses."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DefenseAction:
    clip_radius: float

    def __post_init__(self) -> None:
        if isinstance(self.clip_radius, bool):
            raise TypeError('clip_radius must be a real number')
        try:
            radius = float(self.clip_radius)
        except (TypeError, ValueError) as error:
            raise TypeError('clip_radius must be a real number') from error
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError('clip_radius must be finite and positive')
        object.__setattr__(self, 'clip_radius', radius)
