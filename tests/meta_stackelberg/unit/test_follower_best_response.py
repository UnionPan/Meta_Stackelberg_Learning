import dataclasses
import math

import pytest

from meta_stackelberg.agents import IPMScalePolicy
from meta_stackelberg.feedback import SupportEpisodeFeedback
from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.stackelberg import (
    CandidateIPMBestResponseSolver,
    DefenderCommitment,
    FixedIPMResponseOracle,
)


CANDIDATES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)


def _feedback(value: float) -> SupportEpisodeFeedback:
    return SupportEpisodeFeedback(value, ())


def test_solver_runs_every_sorted_candidate_seed_and_adapts_only_clone() -> None:
    commitment = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))
    initialization = IPMScalePolicy(1.0)
    calls = []

    def factory(received, scale, seed):
        calls.append((received.policy_fingerprint, scale, seed))
        return _feedback(-(scale - 3.0) ** 2 + seed * 0.0)

    result = CandidateIPMBestResponseSolver(reversed(CANDIDATES)).solve(
        commitment=commitment,
        follower_initialization=initialization,
        support_seeds=(103, 101, 102),
        support_factory=factory,
    )

    assert [(scale, seed) for _, scale, seed in calls] == [
        (scale, seed) for scale in CANDIDATES for seed in (101, 102, 103)
    ]
    assert initialization.scale == 1.0
    assert result.initial_follower_snapshot.scale == 1.0
    assert result.adapted_follower_snapshot.scale == 3.0
    assert result.training_steps == len(CANDIDATES) * 3
    assert result.leader_fingerprint_before == result.leader_fingerprint_after


def test_solver_uses_arithmetic_mean_and_smaller_scale_for_exact_tie() -> None:
    commitment = DefenderCommitment.create('same', DefenseAction(1.0, 0.2))
    values = {(1.0, 1): 0.0, (1.0, 2): 2.0, (2.0, 1): 1.0, (2.0, 2): 1.0}
    result = CandidateIPMBestResponseSolver((2.0, 1.0)).solve(
        commitment, IPMScalePolicy(1.0), (2, 1),
        lambda _commitment, scale, seed: _feedback(values[(scale, seed)]),
    )
    assert [record.mean_scalar for record in result.candidate_records] == [1.0, 1.0]
    assert result.adapted_follower_snapshot.scale == 1.0


def test_solver_rejects_invalid_candidates_seeds_and_factory_coordinates() -> None:
    with pytest.raises(ValueError):
        CandidateIPMBestResponseSolver((1.0, 1.0))
    with pytest.raises(ValueError):
        CandidateIPMBestResponseSolver((math.nan,))
    solver = CandidateIPMBestResponseSolver((1.0,))
    commitment = DefenderCommitment.create('x', DefenseAction(1.0))
    with pytest.raises(ValueError, match='seeds'):
        solver.solve(commitment, IPMScalePolicy(1.0), (), lambda *args: _feedback(1.0))
    with pytest.raises(TypeError, match='feedback'):
        solver.solve(commitment, IPMScalePolicy(1.0), (1,), lambda *args: object())


def test_fixed_oracle_has_zero_training_and_preserves_commitment() -> None:
    commitment = DefenderCommitment.create('fixed', DefenseAction(0.1, 0.4))
    result = FixedIPMResponseOracle(5.0).solve(
        commitment=commitment,
        follower_initialization=IPMScalePolicy(1.0),
        support_seeds=(1, 2),
        support_factory=lambda *args: pytest.fail('fixed oracle must not train'),
    )
    assert result.initial_follower_snapshot.scale == 5.0
    assert result.adapted_follower_snapshot.scale == 5.0
    assert result.candidate_records == ()
    assert result.training_steps == 0
    assert result.protocol == 'fixed-ipm-response-v1'
    assert result.leader_fingerprint_before == result.leader_fingerprint_after
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.training_steps = 1  # type: ignore[misc]
