"""Sandbox defender implementations and factories.

Main re-exports in this file:
- ``SandboxDefender``
- ``FedAvgDefender`` / ``KrumDefender`` / ``MultiKrumDefender``
- ``MedianDefender`` / ``ClippedMedianDefender``
- ``GeometricMedianDefender`` / ``TrimmedMeanDefender`` /
  ``PaperNormTrimmedMeanDefender`` / ``FLTrustDefender``
- ``AggregationDefender``
- ``DEFENSE_CHOICES``
- ``create_defender``
- ``supported_defense_types``
"""

from fl_sandbox.config.schema import DefenderSection

from .aggregation import (
    ClippedMedianDefender,
    FLTrustDefender,
    FedAvgDefender,
    GeometricMedianDefender,
    KrumDefender,
    MedianDefender,
    MultiKrumDefender,
    PaperNormTrimmedMeanDefender,
    TrimmedMeanDefender,
)
from .aggregation_runtime import (
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
from .base import SandboxDefender
from .factory import DEFENSE_CHOICES, create_defender, supported_defense_types


def build_defender_config_kwargs(defender: DefenderSection) -> dict[str, object]:
    """Translate defender config into ``SandboxConfig`` keyword arguments."""

    defender_impl = create_defender(defender)
    kwargs = {
        "defense_type": defender_impl.defense_type,
        "krum_attackers": defender.krum_attackers,
        "multi_krum_selected": defender.multi_krum_selected,
        "clipped_median_norm": defender.clipped_median_norm,
        "trimmed_mean_ratio": defender.trimmed_mean_ratio,
        "geometric_median_iters": defender.geometric_median_iters,
        "fltrust_root_size": defender.fltrust_root_size,
    }
    kwargs.update(defender_impl.build_config_kwargs())
    return kwargs


__all__ = [
    "AggregationDefender",
    "ClippedMedianDefender",
    "DEFENSE_CHOICES",
    "FLTrustDefender",
    "FedAvgDefender",
    "GeometricMedianDefender",
    "KrumDefender",
    "MedianDefender",
    "MultiKrumDefender",
    "PaperNormTrimmedMeanDefender",
    "SandboxDefender",
    "TrimmedMeanDefender",
    "build_defender_config_kwargs",
    "clipped_median_aggregate",
    "create_defender",
    "fedavg_aggregate",
    "fltrust_aggregate",
    "geometric_median_aggregate",
    "krum_aggregate",
    "median_aggregate",
    "multi_krum_aggregate",
    "supported_defense_types",
    "trimmed_mean_aggregate",
    "vector_to_weights",
    "weights_to_vector",
]
