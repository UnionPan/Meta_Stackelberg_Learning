import dataclasses
import math

import pytest

from meta_stackelberg.agents import IPMScalePolicySnapshot
from meta_stackelberg.experiments.ipm_best_response import (
    CommitmentResponseCurve,
    E3IPMResponseCurveResult,
    FrozenQueryRecord,
    FrozenResponseEvaluation,
    evaluate_e3_oracle_regret_gate,
)
from meta_stackelberg.security.defenses.actions import DefenseAction
from meta_stackelberg.stackelberg import (
    BestResponseResult,
    CandidateSupportRecord,
    DefenderCommitment,
)


def _evaluation(commitment, scale: float, harm: float) -> FrozenResponseEvaluation:
    return FrozenResponseEvaluation(
        commitment.policy_fingerprint,
        IPMScalePolicySnapshot(scale),
        (FrozenQueryRecord(11, scale, 1.0, 0.5, 1.0 + harm, 0.5, ((0, 1),)),),
    )


def _curve(
    commitment_id: str = 'task-a',
    *,
    adapted_scale: float = 3.0,
    initial_harm: float = 3.0,
    adapted_harm: float = 5.0,
    fixed_harms=((0.5, 2.0), (1.0, 3.0), (3.0, 5.0)),
) -> CommitmentResponseCurve:
    commitment = DefenderCommitment.create(commitment_id, DefenseAction(1.0, 0.0))
    candidates = tuple(
        CandidateSupportRecord(scale, ((1, scale),), scale)
        for scale, _ in fixed_harms
    )
    response = BestResponseResult(
        IPMScalePolicySnapshot(1.0),
        IPMScalePolicySnapshot(adapted_scale),
        candidates,
        len(candidates),
        commitment.policy_fingerprint,
        commitment.policy_fingerprint,
    )
    return CommitmentResponseCurve(
        commitment,
        response,
        _evaluation(commitment, 1.0, initial_harm),
        _evaluation(commitment, adapted_scale, adapted_harm),
        tuple(
            (scale, _evaluation(commitment, scale, harm))
            for scale, harm in fixed_harms
        ),
    )


def test_gate_computes_hand_calculated_oracle_regret_and_is_frozen() -> None:
    curve = _curve()
    result = evaluate_e3_oracle_regret_gate(
        E3IPMResponseCurveResult(True, (curve,), ()),
        required_commitment_ids=('task-a',),
        required_candidate_scales=(0.5, 1.0, 3.0),
        oracle_regret_tolerance=1e-6,
        plateau_tolerance=1e-6,
        strict_improvement_margin=1e-4,
    )

    record = result.commitments[0]
    assert record.oracle_harm == 5.0
    assert record.oracle_scales == (3.0,)
    assert record.oracle_regret == 0.0
    assert record.query_harm_span == 3.0
    assert not record.query_plateau
    assert record.strictly_improved
    assert result.independently_improved_commitments == ('task-a',)
    assert result.passed
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.oracle_harm = 0.0  # type: ignore[misc]


def _gate(*curves, ids=None, scales=(0.5, 1.0, 3.0), **kwargs):
    return evaluate_e3_oracle_regret_gate(
        E3IPMResponseCurveResult(True, tuple(curves), ()),
        required_commitment_ids=ids or tuple(curve.commitment.commitment_id for curve in curves),
        required_candidate_scales=scales,
        **kwargs,
    )


@pytest.mark.parametrize(
    ('ids', 'scales'),
    [
        ((), (0.5,)),
        (('x', 'x'), (0.5,)),
        (('x',), ()),
        (('x',), (1.0, 1.0)),
        (('x',), (math.nan,)),
        (('x',), (True,)),
    ],
)
def test_gate_rejects_invalid_required_coordinates(ids, scales) -> None:
    with pytest.raises((TypeError, ValueError)):
        _gate(_curve('x'), ids=ids, scales=scales)


@pytest.mark.parametrize('name', ['oracle_regret_tolerance', 'plateau_tolerance', 'strict_improvement_margin'])
def test_gate_rejects_invalid_tolerances(name: str) -> None:
    with pytest.raises((TypeError, ValueError), match='tolerance|margin'):
        _gate(_curve(), **{name: -1.0})


def test_gate_rejects_commitment_and_candidate_coverage_mismatches() -> None:
    with pytest.raises(ValueError, match='commitment'):
        _gate(_curve(), ids=('missing',))
    with pytest.raises(ValueError, match='candidate'):
        _gate(_curve(), scales=(0.5, 1.0, 2.0, 3.0))


def test_gate_reports_tied_oracles_and_plateau_boundary() -> None:
    tied = _curve(adapted_scale=0.5, initial_harm=4.999, adapted_harm=5.0,
                  fixed_harms=((0.5, 5.0), (1.0, 4.5), (3.0, 5.0)))
    record = _gate(tied, plateau_tolerance=0.5).commitments[0]
    assert record.oracle_scales == (0.5, 3.0)
    assert record.query_harm_span == 0.5
    assert record.query_plateau

    outside = _curve(adapted_scale=0.5, initial_harm=4.0, adapted_harm=5.0,
                     fixed_harms=((0.5, 5.0), (1.0, 4.499999), (3.0, 5.0)))
    assert not _gate(outside, plateau_tolerance=0.5).commitments[0].query_plateau


def test_plateau_needs_no_local_improvement_when_another_commitment_improves() -> None:
    plateau = _curve('plateau', adapted_scale=0.5, initial_harm=5.0, adapted_harm=5.0,
                     fixed_harms=((0.5, 5.0), (1.0, 5.0), (3.0, 5.0)))
    improving = _curve('improving')
    result = _gate(plateau, improving, ids=('plateau', 'improving'))
    assert result.passed
    assert result.independently_improved_commitments == ('improving',)


def test_gate_returns_scientific_failures_in_stable_order() -> None:
    bad = _curve(adapted_scale=1.0, initial_harm=3.0, adapted_harm=3.0)
    result = _gate(bad)
    assert result.failed_requirements == (
        'one or more follower responses exceed oracle regret tolerance',
        'no commitment strictly improves over initial follower',
    )


def test_gate_rejects_fingerprint_seed_scale_and_harm_inconsistency() -> None:
    base = _curve()
    bad_fingerprint = dataclasses.replace(
        base,
        adapted_query=dataclasses.replace(base.adapted_query, commitment_fingerprint='bad'),
    )
    with pytest.raises(ValueError, match='fingerprint'):
        _gate(bad_fingerprint)

    different_seed_record = dataclasses.replace(base.adapted_query.records[0], seed=99)
    bad_seed = dataclasses.replace(
        base,
        adapted_query=dataclasses.replace(base.adapted_query, records=(different_seed_record,)),
    )
    with pytest.raises(ValueError, match='seed'):
        _gate(bad_seed)

    fixed_scale, fixed_evaluation = base.fixed_queries[0]
    bad_scale = dataclasses.replace(
        base,
        fixed_queries=((fixed_scale, dataclasses.replace(
            fixed_evaluation,
            follower_snapshot=IPMScalePolicySnapshot(2.0),
        )),) + base.fixed_queries[1:],
    )
    with pytest.raises(ValueError, match='candidate.*scale'):
        _gate(bad_scale)

    bad_harm_record = dataclasses.replace(base.adapted_query.records[0], attack_loss=math.inf)
    bad_harm = dataclasses.replace(
        base,
        adapted_query=dataclasses.replace(base.adapted_query, records=(bad_harm_record,)),
    )
    with pytest.raises(ValueError, match='harm'):
        _gate(bad_harm)


def test_gate_rejects_adapted_scale_outside_candidate_set() -> None:
    with pytest.raises(ValueError, match='adapted scale'):
        _gate(_curve(adapted_scale=2.0, adapted_harm=5.0))
