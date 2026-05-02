"""Robust aggregation rules and registry."""

from .base import AggregatorConfig
from .registry import DEFENSE_CHOICES, create_aggregator, supported_aggregators
from .rules import (
    AggregationDefender,
    clipped_median_aggregate,
    fedavg_aggregate,
    fltrust_aggregate,
    geometric_median_aggregate,
    krum_aggregate,
    median_aggregate,
    multi_krum_aggregate,
    trimmed_mean_aggregate,
    vector_to_weights,
    weights_to_vector,
)

__all__ = [
    "AggregationDefender",
    "AggregatorConfig",
    "DEFENSE_CHOICES",
    "clipped_median_aggregate",
    "create_aggregator",
    "fedavg_aggregate",
    "fltrust_aggregate",
    "geometric_median_aggregate",
    "krum_aggregate",
    "median_aggregate",
    "multi_krum_aggregate",
    "supported_aggregators",
    "trimmed_mean_aggregate",
    "vector_to_weights",
    "weights_to_vector",
]
