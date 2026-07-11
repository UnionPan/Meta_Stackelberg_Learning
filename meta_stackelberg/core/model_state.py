"""Immutable model tensors and canonical update-direction operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, eq=False)
class ModelState:
    """A copied, read-only tuple of floating-point model tensors."""

    tensors: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        values = tuple(self.tensors)
        if not values:
            raise ValueError('ModelState requires at least one tensor')
        normalized: list[np.ndarray] = []
        for tensor in values:
            value = np.array(tensor, copy=True)
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError('ModelState tensors must have floating dtypes')
            value.setflags(write=False)
            normalized.append(value)
        object.__setattr__(self, 'tensors', tuple(normalized))

    @classmethod
    def from_tensors(cls, tensors: Iterable[np.ndarray]) -> 'ModelState':
        return cls(tuple(tensors))

    def clone(self) -> 'ModelState':
        return ModelState.from_tensors(np.array(tensor, copy=True) for tensor in self.tensors)

    def vector(self) -> np.ndarray:
        return np.concatenate([tensor.reshape(-1) for tensor in self.tensors]).copy()


def from_vector(vector: np.ndarray, template: ModelState) -> ModelState:
    """Restore a flat vector using a template's shapes and dtypes."""

    flat = np.asarray(vector).reshape(-1)
    expected = sum(tensor.size for tensor in template.tensors)
    if flat.size != expected:
        raise ValueError(f'vector length {flat.size} does not match template length {expected}')
    tensors: list[np.ndarray] = []
    offset = 0
    for target in template.tensors:
        end = offset + target.size
        tensors.append(flat[offset:end].astype(target.dtype, copy=True).reshape(target.shape))
        offset = end
    return ModelState.from_tensors(tensors)


def model_difference(new: ModelState, old: ModelState) -> ModelState:
    """Return the canonical FL delta ``new - old``."""

    _validate_compatible(new, old)
    return ModelState.from_tensors(
        np.subtract(new_tensor, old_tensor, dtype=old_tensor.dtype)
        for new_tensor, old_tensor in zip(new.tensors, old.tensors)
    )


def apply_delta(old: ModelState, delta: ModelState, scale: float = 1.0) -> ModelState:
    """Return ``old + scale * delta`` using the model tensor dtypes."""

    _validate_compatible(old, delta)
    if not np.isfinite(scale):
        raise ValueError('delta scale must be finite')
    return ModelState.from_tensors(
        (old_tensor + float(scale) * delta_tensor).astype(old_tensor.dtype, copy=False)
        for old_tensor, delta_tensor in zip(old.tensors, delta.tensors)
    )


def state_l2_norm(state: ModelState) -> float:
    return float(np.linalg.norm(state.vector().astype(np.float64, copy=False)))


def state_cosine(left: ModelState, right: ModelState) -> float:
    _validate_compatible(left, right)
    left_vector = left.vector().astype(np.float64, copy=False)
    right_vector = right.vector().astype(np.float64, copy=False)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(left_vector, right_vector) / denominator, -1.0, 1.0))


def _validate_compatible(left: ModelState, right: ModelState) -> None:
    if len(left.tensors) != len(right.tensors):
        raise ValueError('ModelState tensor counts do not match')
    for index, (left_tensor, right_tensor) in enumerate(zip(left.tensors, right.tensors)):
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(
                f'ModelState tensor shape mismatch at index {index}: '
                f'{left_tensor.shape} != {right_tensor.shape}'
            )
