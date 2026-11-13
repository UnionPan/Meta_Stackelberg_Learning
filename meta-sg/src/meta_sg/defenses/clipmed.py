"""Design: bound each whole-model update's L2 norm before taking coordinate medians.
Data flow: client updates -> norms and scales -> clipping -> median -> parameter shapes.
Compute norms in float64 to preserve direction for large finite float32 updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.defenses.median import coordinate_median
from meta_sg.federated.state import ModelState, stack_updates


@torch.no_grad()
def clipmed(
    updates: Sequence[ModelState],
    *,
    norm_bound: float = 0.1,
) -> AggregationResult:
    vectors = stack_updates(updates)
    norms = torch.linalg.vector_norm(vectors.to(torch.float64), dim=1)
    scales = norm_bound / norms.clamp_min(norm_bound)
    clipped = (vectors.to(torch.float64) * scales[:, None]).to(vectors.dtype)
    return aggregation_result(
        coordinate_median(clipped), updates[0],
        clipmed_norm_bound=norm_bound,
        clipped_clients=int((norms > norm_bound).sum()),
        minimum_clip_scale=float(scales.min()),
    )
