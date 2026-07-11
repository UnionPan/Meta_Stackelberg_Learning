"""Sample-weighted federated averaging over canonical client deltas."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate


class FedAvg:
    """Aggregate deltas weighted by each client's number of examples."""

    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        values = tuple(updates)
        if not values:
            raise ValueError('FedAvg requires at least one client update')
        template = values[0].delta
        _validate_structures(values, template)
        total_examples = sum(update.num_examples for update in values)
        if total_examples <= 0:
            raise ValueError('FedAvg total number of examples must be positive')

        aggregates: list[np.ndarray] = []
        for tensor_index, target in enumerate(template.tensors):
            accumulator = np.zeros(target.shape, dtype=np.float64)
            for update in values:
                accumulator += (
                    update.delta.tensors[tensor_index].astype(np.float64, copy=False)
                    * update.num_examples
                )
            aggregates.append((accumulator / total_examples).astype(target.dtype))
        return ModelState.from_tensors(aggregates)


def _validate_structures(updates: tuple[ClientUpdate, ...], template: ModelState) -> None:
    expected_shapes = tuple(tensor.shape for tensor in template.tensors)
    for update in updates[1:]:
        shapes = tuple(tensor.shape for tensor in update.delta.tensors)
        if shapes != expected_shapes:
            raise ValueError(
                f'client update structure {shapes} does not match expected structure {expected_shapes}'
            )
