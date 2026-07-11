"""A versioned scalar policy for the IPM follower."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class IPMScalePolicySnapshot:
    scale: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, 'scale', _positive_finite(self.scale))


class IPMScalePolicy:
    def __init__(self, scale: float) -> None:
        self._scale = _positive_finite(scale)

    @property
    def scale(self) -> float:
        return self._scale

    def act(self, observation: Any = None) -> float:
        del observation
        return self._scale

    def snapshot(self) -> IPMScalePolicySnapshot:
        return IPMScalePolicySnapshot(self._scale)

    def restore(self, snapshot: IPMScalePolicySnapshot) -> None:
        if not isinstance(snapshot, IPMScalePolicySnapshot):
            raise TypeError('snapshot must be an IPMScalePolicySnapshot')
        if snapshot.schema_version != 1:
            raise ValueError('unknown snapshot schema_version')
        self._scale = snapshot.scale

    def clone(self) -> IPMScalePolicy:
        return self.from_snapshot(self.snapshot())

    @classmethod
    def from_snapshot(cls, snapshot: IPMScalePolicySnapshot) -> IPMScalePolicy:
        policy = cls(snapshot.scale)
        policy.restore(snapshot)
        return policy


def _positive_finite(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError('scale must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError('scale must be a real number') from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError('scale must be finite and positive')
    return result
