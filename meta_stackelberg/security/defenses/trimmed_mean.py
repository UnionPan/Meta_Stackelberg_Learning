"""Equal-client coordinate-wise trimmed mean aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate


@dataclass(frozen=True)
class TrimmingSummary:
    client_count: int
    trim_ratio: float
    per_tail_trim_count: int
    retained_count: int


class CoordinateTrimmedMean:
    def __init__(self, trim_ratio: float) -> None:
        self.trim_ratio = _validated_trim_ratio(trim_ratio)

    def summarize(self, client_count: int) -> TrimmingSummary:
        if not isinstance(client_count, Integral) or isinstance(client_count, bool):
            raise TypeError('client_count must be a non-bool integer')
        count = int(client_count)
        if count <= 0:
            raise ValueError('client_count must be positive')
        trim_count = math.floor(count * self.trim_ratio)
        retained = count - 2 * trim_count
        if retained <= 0:
            raise ValueError('trim configuration must retain at least one update')
        return TrimmingSummary(count, self.trim_ratio, trim_count, retained)

    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        values = _validated_updates(tuple(updates))
        summary = self.summarize(len(values))
        start = summary.per_tail_trim_count
        stop = len(values) - summary.per_tail_trim_count
        return ModelState.from_tensors(
            np.mean(
                np.sort(
                    np.stack([update.delta.tensors[index] for update in values]),
                    axis=0,
                )[start:stop],
                axis=0,
                dtype=np.float64,
            ).astype(template.dtype)
            for index, template in enumerate(values[0].delta.tensors)
        )


def _validated_trim_ratio(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError('trim_ratio must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError('trim_ratio must be a real number') from error
    if not math.isfinite(result):
        raise ValueError('trim_ratio must be finite')
    if result < 0.0 or result >= 0.5:
        raise ValueError('trim_ratio must be within [0, 0.5)')
    return result


def _validated_updates(updates: tuple[ClientUpdate, ...]) -> tuple[ClientUpdate, ...]:
    if not updates:
        raise ValueError('CoordinateTrimmedMean requires at least one client update')
    expected_shapes = tuple(tensor.shape for tensor in updates[0].delta.tensors)
    expected_dtypes = tuple(tensor.dtype for tensor in updates[0].delta.tensors)
    for update in updates:
        shapes = tuple(tensor.shape for tensor in update.delta.tensors)
        dtypes = tuple(tensor.dtype for tensor in update.delta.tensors)
        if shapes != expected_shapes:
            raise ValueError('client update structure does not match trimmed mean template')
        if dtypes != expected_dtypes:
            raise ValueError('client update dtype does not match trimmed mean template')
        if any(not np.all(np.isfinite(tensor)) for tensor in update.delta.tensors):
            raise ValueError('client update must contain only finite values')
    return updates
