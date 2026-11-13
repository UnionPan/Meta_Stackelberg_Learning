"""Design: preserve the equal-client-weight baseline used in existing experiments.
Data flow: client updates -> flattened matrix -> client mean -> parameter shapes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from meta_sg.defenses._common import AggregationResult, aggregation_result
from meta_sg.federated.state import ModelState, stack_updates


@torch.no_grad()
def fedavg(updates: Sequence[ModelState]) -> AggregationResult:
    vectors = stack_updates(updates)
    return aggregation_result(vectors.mean(dim=0), updates[0])
