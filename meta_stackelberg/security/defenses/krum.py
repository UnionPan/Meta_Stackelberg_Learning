"""Deterministic single-Krum robust aggregation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate


class Krum:
    def __init__(self, byzantine_count: int) -> None:
        if (
            isinstance(byzantine_count, bool)
            or not isinstance(byzantine_count, int)
            or byzantine_count < 0
        ):
            raise ValueError('byzantine_count must be a non-negative integer')
        self.byzantine_count = byzantine_count

    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        values = tuple(updates)
        count = len(values)
        if count <= 2 * self.byzantine_count + 2:
            raise ValueError('Krum requires n > 2f + 2 updates')
        vectors = tuple(
            update.delta.vector().astype(np.float64, copy=False)
            for update in values
        )
        if any(vector.shape != vectors[0].shape for vector in vectors):
            raise ValueError('Krum update structures do not match')
        neighbor_count = count - self.byzantine_count - 2
        scores = []
        for index, vector in enumerate(vectors):
            distances = []
            for other_index, other in enumerate(vectors):
                if other_index == index:
                    continue
                difference = vector - other
                distance = float(np.dot(difference, difference))
                if not math.isfinite(distance):
                    raise ValueError('Krum distance must be finite')
                distances.append(distance)
            scores.append(sum(sorted(distances)[:neighbor_count]))
        selected = min(range(count), key=lambda index: (scores[index], index))
        return values[selected].delta.clone()
