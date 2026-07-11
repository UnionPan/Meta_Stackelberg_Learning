"""Canonical norm-bounding followed by coordinate-wise trimmed mean."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.security.defenses.clipping import (
    ClippedAggregator,
    ClippingSummary,
    _clip_client_updates,
)
from meta_stackelberg.security.defenses.trimmed_mean import (
    CoordinateTrimmedMean,
    TrimmingSummary,
)


@dataclass(frozen=True)
class DefenseAggregationSummary:
    clipping: ClippingSummary
    trimming: TrimmingSummary


class ClippedTrimmedMean:
    def __init__(self, clip_radius: float, trim_ratio: float) -> None:
        self.action = DefenseAction(clip_radius, trim_ratio)
        self._trimmed_mean = CoordinateTrimmedMean(self.action.trim_ratio)

    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState:
        clipped = _clip_client_updates(updates, self.action.clip_radius)
        return self._trimmed_mean.aggregate(clipped)

    def summarize(self, updates: Sequence[ClientUpdate]) -> DefenseAggregationSummary:
        values = tuple(updates)
        clipping = ClippedAggregator(
            CoordinateTrimmedMean(0.0),
            self.action.clip_radius,
        ).summarize(values)
        trimming = self._trimmed_mean.summarize(len(values))
        return DefenseAggregationSummary(clipping, trimming)
