import numpy as np

from meta_stackelberg.agents import IPMScalePolicy
from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.stackelberg import CandidateIPMBestResponseSolver, DefenderCommitment
from meta_stackelberg.experiments.ipm_best_response import (
    CANDIDATE_SCALES,
    QUERY_SEEDS,
    SUPPORT_SEEDS,
    evaluate_frozen_ipm_response,
    make_ipm_support_feedback,
)


def test_real_support_is_fresh_finite_and_does_not_need_an_evaluator() -> None:
    commitment = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))
    first = make_ipm_support_feedback(commitment, 3.0, SUPPORT_SEEDS[0])
    replay = make_ipm_support_feedback(commitment, 3.0, SUPPORT_SEEDS[0])

    assert np.isfinite(first.scalar)
    assert first == replay
    assert len(first.records) == 8


def test_solver_and_frozen_query_preserve_leader_follower_and_replay() -> None:
    commitment = DefenderCommitment.create('weak', DefenseAction(10.0, 0.0))
    initial = IPMScalePolicy(1.0)
    result = CandidateIPMBestResponseSolver(CANDIDATE_SCALES).solve(
        commitment, initial, SUPPORT_SEEDS, make_ipm_support_feedback,
    )
    first = evaluate_frozen_ipm_response(commitment, result.adapted_follower_snapshot, QUERY_SEEDS)
    replay = evaluate_frozen_ipm_response(commitment, result.adapted_follower_snapshot, QUERY_SEEDS)

    assert set(SUPPORT_SEEDS).isdisjoint(QUERY_SEEDS)
    assert initial.scale == 1.0
    assert commitment.verify() is None
    assert first == replay
    assert len(first.records) == len(QUERY_SEEDS)
    assert all(record.attack_loss > record.clean_loss for record in first.records)
