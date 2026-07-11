"""Immutable paper-aligned untargeted reward records."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PaperDefenderReward:
    scalar: float
    post_defense_loss: float
    source: str = 'paper-untargeted-post-defense-v1'


@dataclass(frozen=True)
class PaperAttackerReward:
    scalar: float
    loss_increase: float
    post_loss_before: float
    post_loss_after: float
    source: str = 'paper-untargeted-post-defense-v1'


def evaluate_paper_untargeted_rewards(
    *,
    post_loss_before: float,
    post_loss_after: float,
) -> tuple[PaperDefenderReward, PaperAttackerReward]:
    before = _loss(post_loss_before, 'post_loss_before')
    after = _loss(post_loss_after, 'post_loss_after')
    increase = after - before
    return (
        PaperDefenderReward(-after, after),
        PaperAttackerReward(increase, increase, before, after),
    )


def _loss(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result
