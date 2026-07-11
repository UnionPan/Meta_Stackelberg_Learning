from dataclasses import replace
import math

import numpy as np
import pytest

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.experiments.defense_response_matrix import (
    DefenseGridPoint,
    MatrixGateThresholds,
    RawDefenseObservation,
    evaluate_defense_response_matrix,
    evaluate_task_matrix_gate,
)
from meta_stackelberg.security.defenses.actions import DefenseAction


def _observation(
    seed: int,
    grid_point: DefenseGridPoint,
    branch: str,
    *,
    task_id: str = 'fixture',
) -> RawDefenseObservation:
    attack_offset = grid_point.clip_radius + grid_point.trim_ratio if branch == 'attack' else 0.0
    return RawDefenseObservation(
        task_id=task_id,
        seed=seed,
        grid_point=grid_point,
        branch=branch,
        final_clean_loss=0.2 + attack_offset,
        final_clean_accuracy=1.0,
        attack_metric=0.1 + attack_offset,
        aggregate_norms=(grid_point.clip_radius, grid_point.clip_radius / 2.0),
        clipped_fractions=(0.5, 0.25),
        per_tail_trim_counts=(int(grid_point.trim_ratio > 0.0),) * 2,
        retained_counts=(3, 3),
        sampled_clients=((0, 1, 2), (1, 2, 3)),
        final_random_snapshot=RandomSource(seed).capture(),
        final_model_vector=np.array([grid_point.clip_radius, attack_offset], dtype=np.float32),
    )


def test_grid_point_validates_and_orders_two_physical_actions() -> None:
    point = DefenseGridPoint(clip_radius=1.0, trim_ratio=0.2)

    assert point.action == DefenseAction(1.0, 0.2)
    assert DefenseGridPoint(0.5, 0.4) < point


def test_raw_observation_copies_and_freezes_final_model() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)
    observation = replace(
        _observation(1, DefenseGridPoint(1.0, 0.2), 'attack'),
        final_model_vector=source,
    )

    source[0] = 9.0

    np.testing.assert_array_equal(observation.final_model_vector, np.array([1.0, 2.0]))
    assert not observation.final_model_vector.flags.writeable


@pytest.mark.parametrize(
    'changes',
    [
        {'task_id': ''},
        {'branch': 'invalid'},
        {'final_clean_loss': math.nan},
        {'final_clean_accuracy': 1.1},
        {'attack_metric': math.inf},
        {'aggregate_norms': (-1.0, 0.1)},
        {'clipped_fractions': (1.1, 0.0)},
        {'per_tail_trim_counts': (-1, 0)},
        {'retained_counts': (0, 3)},
        {'sampled_clients': ((0, 0), (1, 2))},
        {'final_model_vector': np.array([math.nan], dtype=np.float32)},
        {'aggregate_norms': (0.1,)},
    ],
)
def test_raw_observation_rejects_invalid_fields(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_observation(1, DefenseGridPoint(1.0, 0.2), 'clean'), **changes)


def test_runner_visits_complete_grid_and_independent_reference_in_order() -> None:
    calls = []

    def factory(seed, point, branch):
        calls.append(('grid', seed, point, branch))
        return _observation(seed, point, branch)

    def reference(seed, branch):
        calls.append(('reference', seed, branch))
        return _observation(seed, DefenseGridPoint(2.0, 0.0), branch)

    matrix = evaluate_defense_response_matrix(
        task_id='fixture',
        seeds=(11, 12),
        clip_radii=(0.5, 2.0),
        trim_ratios=(0.0, 0.2),
        observation_factory=factory,
        reference_factory=reference,
        evaluation_protocol='loss-harm-v1',
        attack_metric_direction='higher_is_worse',
    )

    assert len(matrix.points) == 16
    assert len(matrix.references) == 4
    assert calls[:2] == [('reference', 11, 'clean'), ('reference', 11, 'attack')]
    assert calls[2:4] == [
        ('grid', 11, DefenseGridPoint(0.5, 0.0), 'clean'),
        ('grid', 11, DefenseGridPoint(0.5, 0.0), 'attack'),
    ]
    assert matrix.points[0].clip_cost == pytest.approx(math.log(4.0))
    assert matrix.points[0].trim_cost == 0.0
    assert matrix.points[1].attack_harm == pytest.approx(0.5)


@pytest.mark.parametrize(
    ('clips', 'trims', 'seeds'),
    [
        ((), (0.0,), (1,)),
        ((1.0, 0.5), (0.0,), (1,)),
        ((0.5, 0.5), (0.0,), (1,)),
        ((0.5,), (0.2, 0.0), (1,)),
        ((0.5,), (0.0, 0.0), (1,)),
        ((0.5,), (0.0,), ()),
        ((0.5,), (0.0,), (1, 1)),
    ],
)
def test_runner_rejects_invalid_grids(clips, trims, seeds) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_defense_response_matrix(
            task_id='fixture', seeds=seeds, clip_radii=clips, trim_ratios=trims,
            observation_factory=lambda s, p, b: _observation(s, p, b),
            reference_factory=lambda s, b: _observation(s, DefenseGridPoint(1.0, 0.0), b),
            evaluation_protocol='v1', attack_metric_direction='higher_is_worse',
        )


@pytest.mark.parametrize('mismatch', ['coordinate', 'samples', 'snapshot', 'shape', 'horizon'])
def test_runner_rejects_unmatched_factory_results(mismatch: str) -> None:
    def factory(seed, point, branch):
        result = _observation(seed, point, branch)
        if branch == 'attack' and mismatch == 'coordinate':
            return replace(result, grid_point=DefenseGridPoint(2.0, point.trim_ratio))
        if branch == 'attack' and mismatch == 'samples':
            return replace(result, sampled_clients=((9, 8, 7), (1, 2, 3)))
        if branch == 'attack' and mismatch == 'snapshot':
            return replace(result, final_random_snapshot=RandomSource(seed + 1).capture())
        if branch == 'attack' and mismatch == 'shape':
            return replace(result, final_model_vector=np.array([1.0], dtype=np.float32))
        if branch == 'attack' and mismatch == 'horizon':
            return replace(
                result,
                aggregate_norms=(1.0,), clipped_fractions=(0.0,),
                per_tail_trim_counts=(0,), retained_counts=(3,),
                sampled_clients=((0, 1, 2),),
            )
        return result

    with pytest.raises(ValueError):
        evaluate_defense_response_matrix(
            task_id='fixture', seeds=(1,), clip_radii=(1.0,), trim_ratios=(0.0,),
            observation_factory=factory,
            reference_factory=lambda s, b: _observation(s, DefenseGridPoint(1.0, 0.0), b),
            evaluation_protocol='v1', attack_metric_direction='higher_is_worse',
        )


def test_task_gate_returns_measured_spans_and_active_cell_ratio() -> None:
    matrix = evaluate_defense_response_matrix(
        task_id='fixture', seeds=(1, 2), clip_radii=(0.5, 2.0),
        trim_ratios=(0.0, 0.2),
        observation_factory=lambda s, p, b: _observation(s, p, b),
        reference_factory=lambda s, b: _observation(s, DefenseGridPoint(2.0, 0.0), b),
        evaluation_protocol='v1', attack_metric_direction='higher_is_worse',
    )
    result = evaluate_task_matrix_gate(
        matrix,
        MatrixGateThresholds(1e-6, 1e-4, 1e-5, 0.25),
    )

    assert result.passed
    assert result.aggregate_span > 0.0
    assert result.metric_span > 0.0
    assert result.active_cell_ratio == 1.0
    assert result.clip_dimension_active
    assert result.trim_dimension_active


def test_task_gate_only_compares_adjacent_grid_cells() -> None:
    def factory(seed, point, branch):
        base = _observation(seed, point, branch)
        metric = point.clip_radius * 6e-6 if branch == 'attack' else 0.0
        return replace(
            base,
            attack_metric=metric,
            aggregate_norms=(1.0, 1.0),
        )

    matrix = evaluate_defense_response_matrix(
        task_id='fixture', seeds=(1,), clip_radii=(1.0, 2.0, 3.0),
        trim_ratios=(0.0,), observation_factory=factory,
        reference_factory=lambda s, b: _observation(s, DefenseGridPoint(3.0, 0.0), b),
        evaluation_protocol='v1', attack_metric_direction='higher_is_worse',
    )
    result = evaluate_task_matrix_gate(
        matrix,
        MatrixGateThresholds(0.0, 0.0, 1e-5, 0.0),
    )

    assert not result.clip_dimension_active
    assert result.active_cell_ratio == 0.0
