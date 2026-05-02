"""Aggregator registry used by benchmark configuration."""

from __future__ import annotations

from .rules import DEFENSE_CHOICES, AggregationDefender


def supported_aggregators() -> tuple[str, ...]:
    return DEFENSE_CHOICES


def create_aggregator(defender_config) -> AggregationDefender:
    defense_type = str(getattr(defender_config, "type", getattr(defender_config, "defense_type", "fedavg")))
    if defense_type not in DEFENSE_CHOICES:
        supported = ", ".join(supported_aggregators())
        raise ValueError(f"Unsupported defense type: {defense_type}. Supported defense types: {supported}")
    return AggregationDefender(
        defense_type=defense_type,
        krum_attackers=getattr(defender_config, "krum_attackers", 1),
        multi_krum_selected=getattr(defender_config, "multi_krum_selected", None),
        clipped_median_norm=getattr(defender_config, "clipped_median_norm", 2.0),
        trimmed_mean_ratio=getattr(defender_config, "trimmed_mean_ratio", 0.2),
        geometric_median_iters=getattr(defender_config, "geometric_median_iters", 10),
    )
