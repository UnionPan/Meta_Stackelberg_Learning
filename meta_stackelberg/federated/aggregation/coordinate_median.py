"""Coordinate-wise median aggregation over canonical client deltas."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate


class CoordinateMedian:
    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        values = tuple(updates)
        if not values:
            raise ValueError('CoordinateMedian requires at least one client update')
        template = values[0].delta
        expected_shapes = tuple(tensor.shape for tensor in template.tensors)
        expected_dtypes = tuple(tensor.dtype for tensor in template.tensors)
        for update in values[1:]:
            shapes = tuple(tensor.shape for tensor in update.delta.tensors)
            dtypes = tuple(tensor.dtype for tensor in update.delta.tensors)
            if shapes != expected_shapes:
                raise ValueError('client update structure does not match median template')
            if dtypes != expected_dtypes:
                raise ValueError('client update dtype does not match median template')
        return ModelState.from_tensors(
            np.median(
                np.stack([update.delta.tensors[index] for update in values]),
                axis=0,
            ).astype(template_tensor.dtype, copy=False)
            for index, template_tensor in enumerate(template.tensors)
        )
