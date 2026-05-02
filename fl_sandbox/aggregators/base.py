"""Typed configuration shared by aggregator implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AggregatorConfig:
    """Runtime parameters for robust aggregation rules."""

    defense_type: str = "fedavg"
    krum_attackers: int = 1
    multi_krum_selected: Optional[int] = None
    clipped_median_norm: float = 2.0
    trimmed_mean_ratio: float = 0.2
    geometric_median_iters: int = 10
