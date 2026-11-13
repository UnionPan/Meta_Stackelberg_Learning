"""Design: share tensor conversion and result packing across defense methods.
Data flow: client tensor tuples -> [clients, parameters] matrix;
aggregated vector -> original parameter shapes and method diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from meta_sg.federated.state import ModelState, unflatten_update


AggregationResult = tuple[ModelState, dict[str, Any]]


def aggregation_result(
    vector: torch.Tensor,
    reference: ModelState,
    **diagnostics: Any,
) -> AggregationResult:
    return unflatten_update(vector, reference), {'aggregate_norm': float(vector.norm()), **diagnostics}
