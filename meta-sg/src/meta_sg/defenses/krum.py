"""Design: minimize the sum of squared distances to the n-f-2 nearest neighbors.
Data flow: update matrix -> pairwise distances -> scores -> selected update.
Multi-Krum reuses the scores; Krum breaks ties by selecting the first client."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, stack_updates


def krum_scores(vectors: torch.Tensor, byzantine_bound: int) -> torch.Tensor:
    n = len(vectors)
    if n < 2 * byzantine_bound + 3:
        raise ValueError('Krum requires n >= 2f+3')
    distances = torch.cdist(vectors, vectors).square()
    distances.fill_diagonal_(torch.inf)
    return distances.topk(n - byzantine_bound - 2, dim=1, largest=False).values.sum(dim=1)


@torch.no_grad()
def krum(
    updates: Sequence[ModelState],
    *,
    byzantine_bound: int = 4,
) -> AggregationResult:
    vectors = stack_updates(updates)
    chosen = int(krum_scores(vectors, byzantine_bound).argmin())
    return aggregation_result(vectors[chosen], updates[0], selected_indices=[chosen])
