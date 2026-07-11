from dataclasses import replace
import math

import numpy as np
import pytest

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.experiments.defense_response_surface import (
    ClipControllabilityGate,
    RawClipObservation,
    evaluate_clip_response_surface,
)


def _observation(seed: int, radius: float, branch: str) -> RawClipObservation:
    branch_loss = 0.2 if branch == 'clean' else 0.2 + radius
    return RawClipObservation(
        seed=seed,
        radius=radius,
        branch=branch,
        final_clean_loss=branch_loss,
        final_clean_accuracy=1.0,
        aggregate_norms=(radius, radius / 2.0),
        clipped_client_fractions=(0.5, 0.25),
        sampled_clients=((0, 1), (1, 2)),
        final_random_snapshot=RandomSource(seed).capture(),
        final_model_vector=np.array([radius, branch_loss], dtype=np.float32),
    )


def test_runner_visits_every_point_in_canonical_order() -> None:
    calls: list[tuple[int, float, str]] = []

    def factory(seed: int, radius: float, branch: str) -> RawClipObservation:
        calls.append((seed, radius, branch))
        return _observation(seed, radius, branch)

    surface = evaluate_clip_response_surface(
        radii=(0.25, 1.0),
        seeds=(11, 12),
        observation_factory=factory,
        task_fingerprint='fixture-v1',
        evaluation_protocol='clean-loss-v1',
    )

    assert calls == [
        (11, 0.25, 'clean'),
        (11, 0.25, 'attack'),
        (11, 1.0, 'clean'),
        (11, 1.0, 'attack'),
        (12, 0.25, 'clean'),
        (12, 0.25, 'attack'),
        (12, 1.0, 'clean'),
        (12, 1.0, 'attack'),
    ]
    assert surface.reference_radius == 1.0
    assert len(surface.points) == 8
    assert surface.points[0].defense_cost == pytest.approx(math.log(4.0))
    assert surface.points[0].attack_harm == 0.0
    assert surface.points[1].attack_harm == pytest.approx(0.25)


@pytest.mark.parametrize(
    ('radii', 'seeds'),
    [
        ((), (1,)),
        ((0.0,), (1,)),
        ((1.0, 0.5), (1,)),
        ((0.5, 0.5), (1,)),
        ((0.5,), ()),
        ((0.5,), (1, 1)),
        ((0.5,), (True,)),
    ],
)
def test_runner_rejects_invalid_grids(radii, seeds) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_clip_response_surface(
            radii=radii,
            seeds=seeds,
            observation_factory=_observation,
            task_fingerprint='fixture-v1',
            evaluation_protocol='clean-loss-v1',
        )


@pytest.mark.parametrize('field', ['task_fingerprint', 'evaluation_protocol'])
def test_runner_rejects_empty_identifiers(field: str) -> None:
    arguments = {
        'radii': (0.5,),
        'seeds': (1,),
        'observation_factory': _observation,
        'task_fingerprint': 'fixture-v1',
        'evaluation_protocol': 'clean-loss-v1',
    }
    arguments[field] = ''
    with pytest.raises(ValueError):
        evaluate_clip_response_surface(**arguments)


def test_raw_observation_rejects_non_finite_metrics() -> None:
    with pytest.raises(ValueError, match='finite'):
        replace(_observation(1, 0.5, 'clean'), final_clean_loss=np.nan)


def test_raw_observation_copies_and_freezes_final_model_vector() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)
    observation = replace(_observation(1, 0.5, 'clean'), final_model_vector=source)

    source[0] = 9.0

    np.testing.assert_array_equal(observation.final_model_vector, np.array([1.0, 2.0]))
    assert not observation.final_model_vector.flags.writeable


@pytest.mark.parametrize('mismatch', ['coordinates', 'samples', 'snapshot', 'length'])
def test_runner_rejects_factory_or_matching_mismatch(mismatch: str) -> None:
    def factory(seed: int, radius: float, branch: str) -> RawClipObservation:
        result = _observation(seed, radius, branch)
        if branch == 'attack' and mismatch == 'coordinates':
            return replace(result, radius=radius + 1.0)
        if branch == 'attack' and mismatch == 'samples':
            return replace(result, sampled_clients=((9, 8), (1, 2)))
        if branch == 'attack' and mismatch == 'snapshot':
            return replace(result, final_random_snapshot=RandomSource(seed + 1).capture())
        if branch == 'attack' and mismatch == 'length':
            return replace(result, aggregate_norms=(radius,))
        return result

    with pytest.raises(ValueError):
        evaluate_clip_response_surface(
            radii=(0.5,),
            seeds=(1,),
            observation_factory=factory,
            task_fingerprint='fixture-v1',
            evaluation_protocol='clean-loss-v1',
        )


def test_controllability_gate_uses_precommitted_strict_thresholds() -> None:
    surface = evaluate_clip_response_surface(
        radii=(0.25, 1.0),
        seeds=(1,),
        observation_factory=_observation,
        task_fingerprint='fixture-v1',
        evaluation_protocol='clean-loss-v1',
    )

    assert ClipControllabilityGate().evaluate(surface)
    assert not ClipControllabilityGate(
        aggregate_span_threshold=1.0,
        metric_span_threshold=1.0,
    ).evaluate(surface)
