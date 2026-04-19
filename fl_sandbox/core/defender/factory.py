"""Factory helpers for constructing sandbox defenders."""

from __future__ import annotations

from fl_sandbox.config.schema import DefenderSection

from .aggregation import (
    ClippedMedianDefender,
    FLTrustDefender,
    FedAvgDefender,
    GeometricMedianDefender,
    KrumDefender,
    MedianDefender,
    MultiKrumDefender,
    TrimmedMeanDefender,
)
from .aggregation_runtime import DEFENSE_CHOICES
from .base import SandboxDefender


def supported_defense_types() -> tuple[str, ...]:
    return DEFENSE_CHOICES


def create_defender(defender_config: DefenderSection) -> SandboxDefender:
    defense_type = getattr(defender_config, "type", None)
    if defense_type is None:
        raise ValueError("Defender config is missing required field: type")

    defense_type = str(defense_type)
    if defense_type not in DEFENSE_CHOICES:
        supported = ", ".join(supported_defense_types())
        raise ValueError(f"Unsupported defense type: {defense_type}. Supported defense types: {supported}")
    if defense_type == "fedavg":
        return FedAvgDefender()
    if defense_type == "krum":
        return KrumDefender(krum_attackers=defender_config.krum_attackers)
    if defense_type == "multi_krum":
        return MultiKrumDefender(
            krum_attackers=defender_config.krum_attackers,
            multi_krum_selected=defender_config.multi_krum_selected,
        )
    if defense_type == "median":
        return MedianDefender()
    if defense_type == "clipped_median":
        return ClippedMedianDefender(clipped_median_norm=defender_config.clipped_median_norm)
    if defense_type == "geometric_median":
        return GeometricMedianDefender(geometric_median_iters=defender_config.geometric_median_iters)
    if defense_type == "trimmed_mean":
        return TrimmedMeanDefender(trimmed_mean_ratio=defender_config.trimmed_mean_ratio)
    if defense_type == "fltrust":
        return FLTrustDefender(fltrust_root_size=defender_config.fltrust_root_size)
    raise AssertionError(f"Unreachable defense type branch: {defense_type}")
