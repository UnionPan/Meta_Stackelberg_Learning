"""Design: weight by positive root cosine similarity and normalize to the root norm.
Data flow: client and trusted root updates -> trust weights -> norm scaling ->
weighted sum -> parameter shapes. Zero total trust returns a zero update."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, flatten_update, stack_updates


@torch.no_grad()
def fltrust(
    updates: Sequence[ModelState],
    *,
    root_update: ModelState,
) -> AggregationResult:
    vectors = stack_updates(updates)
    root = flatten_update(root_update).to(vectors)
    root_norm = root.norm()
    norms = torch.linalg.vector_norm(vectors, dim=1)
    denominator = norms * root_norm
    trusts = torch.where(
        denominator > 0,
        (vectors @ root) / denominator.clamp_min(1e-30),
        0.0,
    ).clamp(0, 1)
    total = trusts.sum()
    if float(total) == 0:
        result = torch.zeros_like(root)
    else:
        scales = torch.where(norms > 0, root_norm / norms.clamp_min(1e-30), 0.0)
        result = ((trusts / total * scales)[:, None] * vectors).sum(dim=0)
    return aggregation_result(
        result, updates[0], trust_sum=float(total), zero_trust=float(total) == 0,
    )
