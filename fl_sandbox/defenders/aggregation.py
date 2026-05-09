"""Aggregation-style defender adapters for sandbox orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SandboxDefender


@dataclass
class FedAvgDefender(SandboxDefender):
    name: str = "FedAvg"
    defense_type: str = "fedavg"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {"defense_type": self.defense_type}


@dataclass
class KrumDefender(SandboxDefender):
    krum_attackers: int = 1
    name: str = "Krum"
    defense_type: str = "krum"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "krum_attackers": self.krum_attackers,
        }


@dataclass
class MultiKrumDefender(SandboxDefender):
    krum_attackers: int = 1
    multi_krum_selected: int | None = None
    name: str = "MultiKrum"
    defense_type: str = "multi_krum"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "krum_attackers": self.krum_attackers,
            "multi_krum_selected": self.multi_krum_selected,
        }


@dataclass
class MedianDefender(SandboxDefender):
    name: str = "Median"
    defense_type: str = "median"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {"defense_type": self.defense_type}


@dataclass
class ClippedMedianDefender(SandboxDefender):
    clipped_median_norm: float = 2.0
    name: str = "ClippedMedian"
    defense_type: str = "clipped_median"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "clipped_median_norm": self.clipped_median_norm,
        }


@dataclass
class GeometricMedianDefender(SandboxDefender):
    geometric_median_iters: int = 10
    name: str = "GeometricMedian"
    defense_type: str = "geometric_median"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "geometric_median_iters": self.geometric_median_iters,
        }


@dataclass
class TrimmedMeanDefender(SandboxDefender):
    trimmed_mean_ratio: float = 0.2
    name: str = "TrimmedMean"
    defense_type: str = "trimmed_mean"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "trimmed_mean_ratio": self.trimmed_mean_ratio,
        }


@dataclass
class PaperNormTrimmedMeanDefender(SandboxDefender):
    clipped_median_norm: float = 2.0
    trimmed_mean_ratio: float = 0.2
    name: str = "PaperNormTrimmedMean"
    defense_type: str = "paper_norm_trimmed_mean"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "clipped_median_norm": self.clipped_median_norm,
            "trimmed_mean_ratio": self.trimmed_mean_ratio,
        }


@dataclass
class FLTrustDefender(SandboxDefender):
    fltrust_root_size: int = 100
    name: str = "FLTrust"
    defense_type: str = "fltrust"

    def build_config_kwargs(self) -> dict[str, Any]:
        return {
            "defense_type": self.defense_type,
            "fltrust_root_size": self.fltrust_root_size,
        }
