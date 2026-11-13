"""Design: average the m best updates from one Krum scoring pass, as in existing runs.
Data flow: update matrix -> Krum scores -> stable sort -> top-m mean -> parameter shapes.
Scores are not recomputed after each selection."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.defenses.krum import krum_scores
from meta_sg.federated.state import ModelState, stack_updates


@torch.no_grad()
def multi_krum(
    updates: Sequence[ModelState],
    *,
    byzantine_bound: int = 4,
    selected: int = 16,
) -> AggregationResult:
    if not 1 <= selected <= len(updates):
        raise ValueError('Multi-Krum requires 1 <= selected <= n')
    vectors = stack_updates(updates)
    chosen = krum_scores(vectors, byzantine_bound).argsort(stable=True)[:selected]
    return aggregation_result(
        vectors[chosen].mean(dim=0), updates[0],
        selected_indices=chosen.cpu().tolist(),
    )
