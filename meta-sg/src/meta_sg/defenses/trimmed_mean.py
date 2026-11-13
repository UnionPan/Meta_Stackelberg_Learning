"""Design: discard the f smallest and largest values per coordinate, then average.
Here f is the Byzantine client bound.
Data flow: client updates -> sorted coordinates -> trimmed mean -> parameter shapes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, stack_updates


@torch.no_grad()
def trimmed_mean(
    updates: Sequence[ModelState],
    *,
    byzantine_bound: int = 4,
) -> AggregationResult:
    n, f = len(updates), byzantine_bound
    if n <= 2 * f:
        raise ValueError('trimmed mean requires n > 2f')
    vectors = stack_updates(updates)
    result = vectors.sort(dim=0).values[f:n - f].mean(dim=0)
    return aggregation_result(result, updates[0])
