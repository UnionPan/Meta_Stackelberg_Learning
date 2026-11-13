"""Design: approximate the geometric median with smoothed Weiszfeld iterations.
Data flow: mean initialization -> inverse-distance weights -> weighted estimate ->
iteration limit, zero objective, or relative tolerance. epsilon smooths zero
distances; ftol sets the relative objective-change threshold."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, stack_updates


@torch.no_grad()
def rfa(
    updates: Sequence[ModelState],
    *,
    iterations: int = 4,
    epsilon: float = 1e-5,
    ftol: float = 1e-6,
) -> AggregationResult:
    vectors = stack_updates(updates)
    result = vectors.mean(dim=0)
    objective = torch.linalg.vector_norm(vectors - result, dim=1).mean()
    for iteration in range(iterations):
        distances = torch.linalg.vector_norm(vectors - result, dim=1)
        weights = distances.clamp_min(epsilon).reciprocal()
        weights /= weights.sum()
        candidate = (weights[:, None] * vectors).sum(dim=0)
        next_objective = torch.linalg.vector_norm(vectors - candidate, dim=1).mean()
        improvement = (objective - next_objective).abs()
        result, objective = candidate, next_objective
        objective_value = float(objective)
        if objective_value == 0 or float(improvement) < ftol * objective_value:
            break
    return aggregation_result(
        result, updates[0],
        rfa_iterations=iteration + 1,
        rfa_objective=objective_value,
    )
