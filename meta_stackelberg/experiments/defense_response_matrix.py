"""Matched two-dimensional defense response matrices and controllability gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Callable, Literal

import numpy as np

from meta_stackelberg.core.random_state import RandomSnapshot
from meta_stackelberg.security.defenses.actions import DefenseAction

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


Branch = Literal['clean', 'attack']
MetricDirection = Literal['higher_is_worse', 'lower_is_worse']


@dataclass(frozen=True, order=True)
class DefenseGridPoint:
    clip_radius: float
    trim_ratio: float

    def __post_init__(self) -> None:
        action = DefenseAction(self.clip_radius, self.trim_ratio)
        object.__setattr__(self, 'clip_radius', action.clip_radius)
        object.__setattr__(self, 'trim_ratio', action.trim_ratio)

    @property
    def action(self) -> DefenseAction:
        return DefenseAction(self.clip_radius, self.trim_ratio)


@dataclass(frozen=True)
class RawDefenseObservation:
    task_id: str
    seed: int
    grid_point: DefenseGridPoint
    branch: Branch
    final_clean_loss: float
    final_clean_accuracy: float
    attack_metric: float
    aggregate_norms: tuple[float, ...]
    clipped_fractions: tuple[float, ...]
    per_tail_trim_counts: tuple[int, ...]
    retained_counts: tuple[int, ...]
    sampled_clients: tuple[tuple[int, ...], ...]
    final_random_snapshot: RandomSnapshot
    final_model_vector: np.ndarray

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id must not be empty')
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise TypeError('seed must be a non-bool integer')
        if self.branch not in ('clean', 'attack'):
            raise ValueError("branch must be 'clean' or 'attack'")
        loss = _finite(self.final_clean_loss, 'final_clean_loss')
        accuracy = _finite(self.final_clean_accuracy, 'final_clean_accuracy')
        metric = _finite(self.attack_metric, 'attack_metric')
        if accuracy < 0.0 or accuracy > 1.0:
            raise ValueError('final_clean_accuracy must be within [0, 1]')
        norms = tuple(_nonnegative(value, 'aggregate norm') for value in self.aggregate_norms)
        fractions = tuple(_fraction(value, 'clipped fraction') for value in self.clipped_fractions)
        trims = tuple(_nonnegative_integer(value, 'trim count') for value in self.per_tail_trim_counts)
        retained = tuple(_positive_integer(value, 'retained count') for value in self.retained_counts)
        samples = tuple(_sample_ids(value) for value in self.sampled_clients)
        lengths = {len(norms), len(fractions), len(trims), len(retained), len(samples)}
        if len(lengths) != 1 or not norms:
            raise ValueError('all round observation fields must have the same non-zero length')
        vector = np.array(self.final_model_vector, copy=True)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError('final_model_vector must be a non-empty vector')
        if not np.issubdtype(vector.dtype, np.floating):
            raise TypeError('final_model_vector must have a floating dtype')
        if not np.all(np.isfinite(vector)):
            raise ValueError('final_model_vector must be finite')
        vector.setflags(write=False)
        object.__setattr__(self, 'seed', int(self.seed))
        object.__setattr__(self, 'final_clean_loss', loss)
        object.__setattr__(self, 'final_clean_accuracy', accuracy)
        object.__setattr__(self, 'attack_metric', metric)
        object.__setattr__(self, 'aggregate_norms', norms)
        object.__setattr__(self, 'clipped_fractions', fractions)
        object.__setattr__(self, 'per_tail_trim_counts', trims)
        object.__setattr__(self, 'retained_counts', retained)
        object.__setattr__(self, 'sampled_clients', samples)
        object.__setattr__(self, 'final_model_vector', vector)


@dataclass(frozen=True)
class DefenseResponsePoint:
    observation: RawDefenseObservation
    attack_harm: float
    clip_cost: float
    trim_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'attack_harm', _finite(self.attack_harm, 'attack_harm'))
        object.__setattr__(self, 'clip_cost', _nonnegative(self.clip_cost, 'clip_cost'))
        object.__setattr__(self, 'trim_cost', _nonnegative(self.trim_cost, 'trim_cost'))


@dataclass(frozen=True)
class DefenseResponseMatrix:
    task_id: str
    evaluation_protocol: str
    attack_metric_direction: MetricDirection
    reference_clip_radius: float
    max_trim_ratio: float
    points: tuple[DefenseResponsePoint, ...]
    references: tuple[RawDefenseObservation, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.evaluation_protocol:
            raise ValueError('matrix identifiers must not be empty')
        if self.attack_metric_direction not in ('higher_is_worse', 'lower_is_worse'):
            raise ValueError('invalid attack metric direction')
        if not self.points or not self.references:
            raise ValueError('matrix points and references must not be empty')
        object.__setattr__(self, 'points', tuple(self.points))
        object.__setattr__(self, 'references', tuple(self.references))


DefenseObservationFactory = Callable[[int, DefenseGridPoint, Branch], RawDefenseObservation]
ReferenceFactory = Callable[[int, Branch], RawDefenseObservation]


def evaluate_defense_response_matrix(
    *,
    task_id: str,
    seeds: tuple[int, ...],
    clip_radii: tuple[float, ...],
    trim_ratios: tuple[float, ...],
    observation_factory: DefenseObservationFactory,
    reference_factory: ReferenceFactory,
    evaluation_protocol: str,
    attack_metric_direction: MetricDirection,
) -> DefenseResponseMatrix:
    if not task_id or not evaluation_protocol:
        raise ValueError('matrix identifiers must not be empty')
    if attack_metric_direction not in ('higher_is_worse', 'lower_is_worse'):
        raise ValueError('invalid attack metric direction')
    checked_seeds = _unique_integers(seeds, 'seeds')
    checked_clips = _strictly_increasing_positive(clip_radii, 'clip_radii')
    checked_trims = _strictly_increasing_trims(trim_ratios)
    reference_radius = checked_clips[-1]
    max_trim = checked_trims[-1] if checked_trims[-1] > 0.0 else 1.0
    points: list[DefenseResponsePoint] = []
    references: list[RawDefenseObservation] = []
    for seed in checked_seeds:
        reference_clean = reference_factory(seed, 'clean')
        reference_attack = reference_factory(seed, 'attack')
        _validate_reference(reference_clean, task_id, seed, 'clean')
        _validate_reference(reference_attack, task_id, seed, 'attack')
        _validate_matched(reference_clean, reference_attack)
        references.extend((reference_clean, reference_attack))
        for clip_radius in checked_clips:
            for trim_ratio in checked_trims:
                grid_point = DefenseGridPoint(clip_radius, trim_ratio)
                clean = observation_factory(seed, grid_point, 'clean')
                attack = observation_factory(seed, grid_point, 'attack')
                _validate_grid_observation(clean, task_id, seed, grid_point, 'clean')
                _validate_grid_observation(attack, task_id, seed, grid_point, 'attack')
                _validate_matched(clean, attack)
                raw_harm = attack.attack_metric - clean.attack_metric
                harm = raw_harm if attack_metric_direction == 'higher_is_worse' else -raw_harm
                clip_cost = math.log(reference_radius / clip_radius)
                trim_cost = trim_ratio / max_trim
                points.extend((
                    DefenseResponsePoint(clean, 0.0, clip_cost, trim_cost),
                    DefenseResponsePoint(attack, harm, clip_cost, trim_cost),
                ))
    return DefenseResponseMatrix(
        task_id=task_id,
        evaluation_protocol=evaluation_protocol,
        attack_metric_direction=attack_metric_direction,
        reference_clip_radius=reference_radius,
        max_trim_ratio=checked_trims[-1],
        points=tuple(points),
        references=tuple(references),
    )


@dataclass(frozen=True)
class MatrixGateThresholds:
    aggregate_span: float
    metric_span: float
    active_cell_delta: float
    min_active_cell_ratio: float

    def __post_init__(self) -> None:
        for name in ('aggregate_span', 'metric_span', 'active_cell_delta'):
            value = _nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)
        ratio = _fraction(self.min_active_cell_ratio, 'min_active_cell_ratio')
        object.__setattr__(self, 'min_active_cell_ratio', ratio)


@dataclass(frozen=True)
class TaskMatrixGateResult:
    task_id: str
    passed: bool
    aggregate_span: float
    metric_span: float
    active_cell_ratio: float
    clip_dimension_active: bool
    trim_dimension_active: bool


def evaluate_task_matrix_gate(
    matrix: DefenseResponseMatrix,
    thresholds: MatrixGateThresholds,
) -> TaskMatrixGateResult:
    attack_points = tuple(
        point for point in matrix.points if point.observation.branch == 'attack'
    )
    by_cell: dict[DefenseGridPoint, list[DefenseResponsePoint]] = {}
    for point in attack_points:
        by_cell.setdefault(point.observation.grid_point, []).append(point)
    aggregate_values = {
        cell: float(np.mean([
            value
            for point in points
            for value in point.observation.aggregate_norms
        ]))
        for cell, points in by_cell.items()
    }
    metric_values = {
        cell: float(np.mean([point.attack_harm for point in points]))
        for cell, points in by_cell.items()
    }
    aggregate_span = max(aggregate_values.values()) - min(aggregate_values.values())
    metric_span = max(metric_values.values()) - min(metric_values.values())
    active_cells: set[DefenseGridPoint] = set()
    clip_active = False
    trim_active = False
    cells = tuple(sorted(by_cell))
    clip_values = tuple(sorted({cell.clip_radius for cell in cells}))
    trim_values = tuple(sorted({cell.trim_ratio for cell in cells}))
    for left in cells:
        neighbors: list[tuple[DefenseGridPoint, str]] = []
        clip_index = clip_values.index(left.clip_radius)
        if clip_index + 1 < len(clip_values):
            neighbors.append((
                DefenseGridPoint(clip_values[clip_index + 1], left.trim_ratio),
                'clip',
            ))
        trim_index = trim_values.index(left.trim_ratio)
        if trim_index + 1 < len(trim_values):
            neighbors.append((
                DefenseGridPoint(left.clip_radius, trim_values[trim_index + 1]),
                'trim',
            ))
        for right, dimension in neighbors:
            delta = max(
                abs(aggregate_values[left] - aggregate_values[right]),
                abs(metric_values[left] - metric_values[right]),
            )
            if delta > thresholds.active_cell_delta:
                active_cells.update((left, right))
                clip_active = clip_active or dimension == 'clip'
                trim_active = trim_active or dimension == 'trim'
    active_ratio = len(active_cells) / len(cells)
    passed = (
        aggregate_span > thresholds.aggregate_span
        and metric_span > thresholds.metric_span
        and active_ratio >= thresholds.min_active_cell_ratio
    )
    return TaskMatrixGateResult(
        task_id=matrix.task_id,
        passed=passed,
        aggregate_span=aggregate_span,
        metric_span=metric_span,
        active_cell_ratio=active_ratio,
        clip_dimension_active=clip_active,
        trim_dimension_active=trim_active,
    )


def _validate_grid_observation(observation, task_id, seed, point, branch) -> None:
    if (
        observation.task_id,
        observation.seed,
        observation.grid_point,
        observation.branch,
    ) != (task_id, seed, point, branch):
        raise ValueError('observation coordinates do not match requested grid point')


def _validate_reference(observation, task_id, seed, branch) -> None:
    if (observation.task_id, observation.seed, observation.branch) != (task_id, seed, branch):
        raise ValueError('reference coordinates do not match requested branch')


def _validate_matched(clean: RawDefenseObservation, attack: RawDefenseObservation) -> None:
    if clean.sampled_clients != attack.sampled_clients:
        raise ValueError('clean and attack sampled clients do not match')
    if len(clean.aggregate_norms) != len(attack.aggregate_norms):
        raise ValueError('clean and attack horizons do not match')
    if clean.final_model_vector.shape != attack.final_model_vector.shape:
        raise ValueError('clean and attack final model structures do not match')
    if not _snapshots_equal(clean.final_random_snapshot, attack.final_random_snapshot):
        raise ValueError('clean and attack random snapshots do not match')


def _strictly_increasing_positive(values, name) -> tuple[float, ...]:
    checked = tuple(_positive(value, name) for value in values)
    if not checked:
        raise ValueError(f'{name} must not be empty')
    if any(left >= right for left, right in zip(checked, checked[1:])):
        raise ValueError(f'{name} must be strictly increasing')
    return checked


def _strictly_increasing_trims(values) -> tuple[float, ...]:
    checked = tuple(DefenseAction(1.0, value).trim_ratio for value in values)
    if not checked:
        raise ValueError('trim_ratios must not be empty')
    if any(left >= right for left, right in zip(checked, checked[1:])):
        raise ValueError('trim_ratios must be strictly increasing')
    return checked


def _unique_integers(values, name) -> tuple[int, ...]:
    checked = tuple(values)
    if not checked:
        raise ValueError(f'{name} must not be empty')
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in checked):
        raise TypeError(f'{name} must contain non-bool integers')
    if len(checked) != len(set(checked)):
        raise ValueError(f'{name} must contain unique values')
    return tuple(int(value) for value in checked)


def _sample_ids(values) -> tuple[int, ...]:
    result = _unique_integers(tuple(values), 'sampled client ids')
    if any(value < 0 for value in result):
        raise ValueError('sampled client ids must be non-negative')
    return result


def _finite(value, name) -> float:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{name} must be a real number') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _positive(value, name) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f'{name} must be positive')
    return result


def _nonnegative(value, name) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f'{name} must be non-negative')
    return result


def _fraction(value, name) -> float:
    result = _finite(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f'{name} must be within [0, 1]')
    return result


def _nonnegative_integer(value, name) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f'{name} must be a non-bool integer')
    if value < 0:
        raise ValueError(f'{name} must be non-negative')
    return int(value)


def _positive_integer(value, name) -> int:
    result = _nonnegative_integer(value, name)
    if result <= 0:
        raise ValueError(f'{name} must be positive')
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
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)
