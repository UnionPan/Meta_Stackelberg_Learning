"""Server-visible support proxy for an IPM follower."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from meta_stackelberg.core.model_state import model_difference
from meta_stackelberg.federated.episode import FederatedTrajectory
from meta_stackelberg.federated.types import ClientUpdate, RoundTransition


@dataclass(frozen=True)
class AttackerProxyRecord:
    scalar: float
    opposition: float
    survival: float
    scale_cost: float
    source: str = 'ipm-transition-proxy-v1'


@dataclass(frozen=True)
class SupportEpisodeFeedback:
    scalar: float
    records: tuple[AttackerProxyRecord, ...]
    source: str = 'ipm-support-mean-v1'


class IPMAttackerProxy:
    epsilon = 1e-12
    survival_weight = 0.25
    scale_cost_weight = 0.01

    def evaluate_transition(
        self,
        transition: RoundTransition,
        scale: float,
    ) -> AttackerProxyRecord:
        checked_scale = _positive_finite(scale, 'scale')
        benign = _mean_updates(transition.benign_updates, 'benign updates')
        malicious = _mean_updates(transition.malicious_updates, 'malicious updates')
        deployed = model_difference(
            transition.state_after.global_model,
            transition.state_before.global_model,
        ).vector().astype(np.float64, copy=False)
        _require_finite(deployed)
        opposition = -float(np.dot(deployed, benign)) / (
            float(np.dot(benign, benign)) + self.epsilon
        )
        survival = float(np.dot(deployed, malicious)) / (
            float(np.dot(malicious, malicious)) + self.epsilon
        )
        scale_cost = self.scale_cost_weight * math.log1p(checked_scale**2)
        scalar = opposition + self.survival_weight * survival - scale_cost
        values = (scalar, opposition, survival, scale_cost)
        if any(not math.isfinite(value) for value in values):
            raise ValueError('proxy components must be finite')
        return AttackerProxyRecord(*values)

    def evaluate_trajectory(
        self,
        trajectory: FederatedTrajectory,
        scale: float,
    ) -> SupportEpisodeFeedback:
        records = tuple(
            self.evaluate_transition(transition, scale)
            for transition in trajectory.transitions
        )
        return SupportEpisodeFeedback(
            scalar=float(np.mean([record.scalar for record in records])),
            records=records,
        )


def _mean_updates(updates: tuple[ClientUpdate, ...], name: str) -> np.ndarray:
    if not updates:
        raise ValueError(f'{name} must not be empty')
    vectors = [update.delta.vector().astype(np.float64, copy=False) for update in updates]
    size = vectors[0].size
    if any(vector.size != size for vector in vectors):
        raise ValueError(f'{name} have incompatible model structures')
    result = np.mean(np.stack(vectors), axis=0)
    _require_finite(result)
    return result


def _require_finite(value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError('proxy inputs must be finite')


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a real number') from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return result
