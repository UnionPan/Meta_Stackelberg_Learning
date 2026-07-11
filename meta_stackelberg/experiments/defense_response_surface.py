"""Pure orchestration and records for matched clipping response surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Callable, Literal

import numpy as np

from meta_stackelberg.core.random_state import RandomSnapshot

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


Branch = Literal['clean', 'attack']


@dataclass(frozen=True)
class RawClipObservation:
    seed: int
    radius: float
    branch: Branch
    final_clean_loss: float
    final_clean_accuracy: float
    aggregate_norms: tuple[float, ...]
    clipped_client_fractions: tuple[float, ...]
    sampled_clients: tuple[tuple[int, ...], ...]
    final_random_snapshot: RandomSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise TypeError('seed must be a non-bool integer')
        if self.branch not in ('clean', 'attack'):
            raise ValueError("branch must be 'clean' or 'attack'")
        _positive_finite(self.radius, 'radius')
        _finite(self.final_clean_loss, 'final_clean_loss')
        accuracy = _finite(self.final_clean_accuracy, 'final_clean_accuracy')
        if accuracy < 0.0 or accuracy > 1.0:
            raise ValueError('final_clean_accuracy must be within [0, 1]')
        norms = tuple(float(value) for value in self.aggregate_norms)
        fractions = tuple(float(value) for value in self.clipped_client_fractions)
        samples = tuple(tuple(int(client_id) for client_id in ids) for ids in self.sampled_clients)
        if not norms or len(norms) != len(fractions) or len(norms) != len(samples):
            raise ValueError('round observation lengths must be equal and non-empty')
        if any(not math.isfinite(value) or value < 0.0 for value in norms):
            raise ValueError('aggregate_norms must be finite and non-negative')
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError('clipped_client_fractions must be finite and within [0, 1]')
        object.__setattr__(self, 'seed', int(self.seed))
        object.__setattr__(self, 'radius', float(self.radius))
        object.__setattr__(self, 'final_clean_loss', float(self.final_clean_loss))
        object.__setattr__(self, 'final_clean_accuracy', accuracy)
        object.__setattr__(self, 'aggregate_norms', norms)
        object.__setattr__(self, 'clipped_client_fractions', fractions)
        object.__setattr__(self, 'sampled_clients', samples)


@dataclass(frozen=True)
class ClipResponsePoint:
    observation: RawClipObservation
    attack_harm: float
    defense_cost: float

    def __post_init__(self) -> None:
        _finite(self.attack_harm, 'attack_harm')
        cost = _finite(self.defense_cost, 'defense_cost')
        if cost < 0.0:
            raise ValueError('defense_cost must be non-negative')


@dataclass(frozen=True)
class ClipResponseSurface:
    task_fingerprint: str
    evaluation_protocol: str
    reference_radius: float
    points: tuple[ClipResponsePoint, ...]

    def __post_init__(self) -> None:
        if not self.task_fingerprint or not self.evaluation_protocol:
            raise ValueError('surface identifiers must not be empty')
        _positive_finite(self.reference_radius, 'reference_radius')
        points = tuple(self.points)
        if not points:
            raise ValueError('response surface must contain points')
        object.__setattr__(self, 'points', points)


ObservationFactory = Callable[[int, float, Branch], RawClipObservation]


def evaluate_clip_response_surface(
    *,
    radii: tuple[float, ...],
    seeds: tuple[int, ...],
    observation_factory: ObservationFactory,
    task_fingerprint: str,
    evaluation_protocol: str,
) -> ClipResponseSurface:
    checked_radii = _validate_radii(radii)
    checked_seeds = _validate_seeds(seeds)
    if not task_fingerprint or not evaluation_protocol:
        raise ValueError('surface identifiers must not be empty')
    reference = checked_radii[-1]
    points: list[ClipResponsePoint] = []
    for seed in checked_seeds:
        for radius in checked_radii:
            clean = observation_factory(seed, radius, 'clean')
            attack = observation_factory(seed, radius, 'attack')
            _validate_coordinates(clean, seed, radius, 'clean')
            _validate_coordinates(attack, seed, radius, 'attack')
            _validate_matched(clean, attack)
            cost = math.log(reference / radius)
            harm = attack.final_clean_loss - clean.final_clean_loss
            points.extend((
                ClipResponsePoint(clean, attack_harm=0.0, defense_cost=cost),
                ClipResponsePoint(attack, attack_harm=harm, defense_cost=cost),
            ))
    return ClipResponseSurface(
        task_fingerprint=task_fingerprint,
        evaluation_protocol=evaluation_protocol,
        reference_radius=reference,
        points=tuple(points),
    )


@dataclass(frozen=True)
class ClipControllabilityGate:
    aggregate_span_threshold: float = 1e-6
    metric_span_threshold: float = 1e-4

    def __post_init__(self) -> None:
        for name in ('aggregate_span_threshold', 'metric_span_threshold'):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative')

    def evaluate(self, surface: ClipResponseSurface) -> bool:
        attack_points = tuple(
            point for point in surface.points if point.observation.branch == 'attack'
        )
        if not attack_points:
            return False
        aggregate_values = tuple(
            value
            for point in attack_points
            for value in point.observation.aggregate_norms
        )
        aggregate_span = max(aggregate_values) - min(aggregate_values)
        losses = tuple(point.observation.final_clean_loss for point in attack_points)
        harms = tuple(point.attack_harm for point in attack_points)
        return aggregate_span > self.aggregate_span_threshold and (
            max(losses) - min(losses) > self.metric_span_threshold
            or max(harms) - min(harms) > self.metric_span_threshold
        )


def _validate_radii(values: tuple[float, ...]) -> tuple[float, ...]:
    radii = tuple(_positive_finite(value, 'radius') for value in values)
    if not radii:
        raise ValueError('radii must not be empty')
    if any(left >= right for left, right in zip(radii, radii[1:])):
        raise ValueError('radii must be strictly increasing')
    return radii


def _validate_seeds(values: tuple[int, ...]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError('seeds must not be empty')
    if any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in seeds):
        raise TypeError('seeds must be non-bool integers')
    if len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be unique')
    return tuple(int(seed) for seed in seeds)


def _validate_coordinates(
    observation: RawClipObservation,
    seed: int,
    radius: float,
    branch: Branch,
) -> None:
    if (observation.seed, observation.radius, observation.branch) != (seed, radius, branch):
        raise ValueError('observation coordinates do not match requested point')


def _validate_matched(clean: RawClipObservation, attack: RawClipObservation) -> None:
    if clean.sampled_clients != attack.sampled_clients:
        raise ValueError('clean and attack sampled clients do not match')
    if len(clean.aggregate_norms) != len(attack.aggregate_norms):
        raise ValueError('clean and attack horizons do not match')
    if not _snapshots_equal(clean.final_random_snapshot, attack.final_random_snapshot):
        raise ValueError('clean and attack final random snapshots do not match')


def _positive_finite(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f'{name} must be positive')
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a real number') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> bool:
    if left.python_state != right.python_state:
        return False
    if not _nested_equal(left.numpy_state, right.numpy_state):
        return False
    if left.torch_cpu_state is None or right.torch_cpu_state is None:
        return left.torch_cpu_state is None and right.torch_cpu_state is None
    return torch is not None and bool(torch.equal(left.torch_cpu_state, right.torch_cpu_state))


def _nested_equal(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and bool(
            np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_nested_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)
