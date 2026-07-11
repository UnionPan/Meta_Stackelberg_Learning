"""Stateless global-L2 clipping for canonical client updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from meta_stackelberg.core.model_state import ModelState, state_l2_norm
from meta_stackelberg.federated.protocols import Aggregator
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.actions import DefenseAction


@dataclass(frozen=True)
class ClippingSummary:
    clip_radius: float
    client_count: int
    clipped_client_count: int
    clipped_client_fraction: float
    pre_clip_norm_min: float
    pre_clip_norm_median: float
    pre_clip_norm_max: float


class ClippedAggregator:
    def __init__(self, base: Aggregator, clip_radius: float) -> None:
        if not isinstance(base, Aggregator):
            raise TypeError('base must satisfy Aggregator')
        self.base = base
        self.clip_radius = DefenseAction(clip_radius).clip_radius

    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        return self.base.aggregate(_clip_client_updates(updates, self.clip_radius))

    def summarize(self, updates: Sequence[ClientUpdate]) -> ClippingSummary:
        norms = _validated_norms(tuple(updates))
        clipped_count = sum(norm > self.clip_radius for norm in norms)
        return ClippingSummary(
            clip_radius=self.clip_radius,
            client_count=len(norms),
            clipped_client_count=clipped_count,
            clipped_client_fraction=clipped_count / len(norms),
            pre_clip_norm_min=min(norms),
            pre_clip_norm_median=float(np.median(norms)),
            pre_clip_norm_max=max(norms),
        )


def _validated_norms(updates: tuple[ClientUpdate, ...]) -> tuple[float, ...]:
    if not updates:
        raise ValueError('ClippedAggregator requires at least one client update')
    expected_shapes = tuple(tensor.shape for tensor in updates[0].delta.tensors)
    expected_dtypes = tuple(tensor.dtype for tensor in updates[0].delta.tensors)
    norms: list[float] = []
    for update in updates:
        shapes = tuple(tensor.shape for tensor in update.delta.tensors)
        dtypes = tuple(tensor.dtype for tensor in update.delta.tensors)
        if shapes != expected_shapes:
            raise ValueError('client update structure does not match clipping template')
        if dtypes != expected_dtypes:
            raise ValueError('client update dtype does not match clipping template')
        if any(not np.all(np.isfinite(tensor)) for tensor in update.delta.tensors):
            raise ValueError('client update must contain only finite values')
        norms.append(state_l2_norm(update.delta))
    return tuple(norms)


def _scaled_update(update: ClientUpdate, scale: float) -> ClientUpdate:
    delta = ModelState.from_tensors(
        (tensor * scale).astype(tensor.dtype, copy=False)
        for tensor in update.delta.tensors
    )
    return ClientUpdate(
        client_id=update.client_id,
        delta=delta,
        num_examples=update.num_examples,
        is_malicious=update.is_malicious,
        metadata=update.metadata,
    )


def _clip_client_updates(
    updates: Sequence[ClientUpdate],
    clip_radius: float,
) -> tuple[ClientUpdate, ...]:
    radius = DefenseAction(clip_radius).clip_radius
    values = tuple(updates)
    norms = _validated_norms(values)
    return tuple(
        _scaled_update(
            update,
            min(1.0, radius / norm) if norm > 0.0 else 1.0,
        )
        for update, norm in zip(values, norms)
    )
