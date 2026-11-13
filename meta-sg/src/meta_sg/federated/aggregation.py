"""Design: apply client updates using the supplied weights.
Data flow: global parameters and client deltas -> normalized weights ->
weighted delta -> new global parameters. Zero total weight preserves the model."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.federated.state import ModelState


def aggregate_client_updates(
    current: ModelState,
    updates: Sequence[ModelState],
    *,
    weights: Sequence[float] | None = None,
) -> ModelState:
    if weights is None:
        weights = [1.0] * len(updates)
    weight_tensor = torch.as_tensor(
        weights, dtype=updates[0][0].dtype, device=updates[0][0].device,
    )
    total = weight_tensor.sum()
    if float(total) == 0:
        return tuple(part.clone() for part in current)
    normalized = weight_tensor / total
    aggregated = [
        sum(weight * part for weight, part in zip(normalized, parts, strict=True))
        for parts in zip(*updates, strict=True)
    ]
    return tuple(
        part + update for part, update in zip(current, aggregated, strict=True)
    )
