"""Design: take coordinate medians, averaging the two middle values for even counts.
Data flow: client updates -> sorted coordinates -> middle values -> parameter shapes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, stack_updates


def coordinate_median(vectors: torch.Tensor) -> torch.Tensor:
    sorted_values = vectors.sort(dim=0).values
    n = len(vectors)
    return (sorted_values[(n - 1) // 2] + sorted_values[n // 2]) / 2


@torch.no_grad()
def median(updates: Sequence[ModelState]) -> AggregationResult:
    return aggregation_result(coordinate_median(stack_updates(updates)), updates[0])
