import dataclasses
import math

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState, apply_delta
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.feedback import IPMAttackerProxy
from meta_stackelberg.federated.episode import FederatedTrajectory
from meta_stackelberg.federated.types import ClientUpdate, RoundState, RoundTransition


def _model(values: list[float]) -> ModelState:
    return ModelState.from_tensors((np.asarray(values, dtype=np.float64),))


def _transition(
    *,
    benign: list[float] = [1.0, 0.0],
    malicious: list[float] = [-2.0, 0.0],
    deployed: list[float] = [-0.5, 0.0],
) -> RoundTransition:
    snapshot = RandomSource(1).capture()
    before = RoundState(0, _model([0.0, 0.0]), snapshot)
    delta = _model(deployed)
    after = RoundState(1, apply_delta(before.global_model, delta), snapshot)
    return RoundTransition(
        task_id='ipm',
        state_before=before,
        sampled_clients=(0, 1),
        benign_updates=(ClientUpdate(0, _model(benign), 1),),
        malicious_updates=(ClientUpdate(1, _model(malicious), 1, True),),
        aggregate_delta=_model([999.0, 999.0]),
        state_after=after,
        private_diagnostics={'forbidden': float('nan')},
    )


def test_proxy_components_match_hand_calculation_and_ignore_private_delta() -> None:
    record = IPMAttackerProxy().evaluate_transition(_transition(), scale=1.0)

    assert record.opposition == pytest.approx(0.5)
    assert record.survival == pytest.approx(0.25)
    assert record.scale_cost == pytest.approx(0.01 * math.log(2.0))
    assert record.scalar == pytest.approx(0.5 + 0.25 * 0.25 - record.scale_cost)
    assert record.source == 'ipm-transition-proxy-v1'
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.scalar = 0.0  # type: ignore[misc]


def test_proxy_uses_model_difference_and_handles_zero_vectors() -> None:
    record = IPMAttackerProxy().evaluate_transition(
        _transition(benign=[0.0, 0.0], malicious=[0.0, 0.0], deployed=[2.0, 3.0]),
        scale=2.0,
    )
    assert record.opposition == 0.0
    assert record.survival == 0.0
    assert record.scalar == pytest.approx(-0.01 * math.log1p(4.0))


def test_proxy_requires_benign_and_malicious_updates() -> None:
    transition = _transition()
    for benign, malicious in (((), transition.malicious_updates), (transition.benign_updates, ())):
        altered = dataclasses.replace(transition, benign_updates=benign, malicious_updates=malicious)
        with pytest.raises(ValueError, match='updates'):
            IPMAttackerProxy().evaluate_transition(altered, 1.0)


def test_proxy_rejects_nonfinite_update_values_and_invalid_scale() -> None:
    with pytest.raises(ValueError, match='finite'):
        IPMAttackerProxy().evaluate_transition(_transition(benign=[math.nan, 0.0]), 1.0)
    with pytest.raises(ValueError, match='scale'):
        IPMAttackerProxy().evaluate_transition(_transition(), math.inf)


def test_trajectory_feedback_is_arithmetic_mean_and_immutable() -> None:
    first = _transition(deployed=[-0.5, 0.0])
    second_before = first.state_after
    delta = _model([-1.0, 0.0])
    second = dataclasses.replace(
        _transition(deployed=[-1.0, 0.0]),
        state_before=second_before,
        state_after=RoundState(2, apply_delta(second_before.global_model, delta), second_before.random_snapshot),
    )
    trajectory = FederatedTrajectory('ipm', first.state_before, (first, second), second.state_after)
    feedback = IPMAttackerProxy().evaluate_trajectory(trajectory, 1.0)

    assert feedback.scalar == pytest.approx(sum(item.scalar for item in feedback.records) / 2)
    assert len(feedback.records) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        feedback.scalar = 0.0  # type: ignore[misc]
