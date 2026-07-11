import numpy as np
import pytest

from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    QueryPolicyEvidence,
    ScientificGateThresholds,
    evaluate_meta_sg_scientific_gate,
    evaluate_frozen_policy,
    evaluate_frozen_pair,
)


class FrozenPolicy:
    def __init__(self):
        self.version = 0

    def fingerprint(self):
        return str(self.version)


def _evidence(label, objective, action):
    return QueryPolicyEvidence(
        label=label,
        query_seeds=(101, 102),
        mean_objective=objective,
        action_trajectory=(tuple(action), tuple(action)),
        policy_fingerprint='frozen',
    )


def test_frozen_query_evaluator_rejects_seed_leakage_and_policy_mutation() -> None:
    with pytest.raises(ValueError, match='disjoint'):
        QueryEvidencePlan(support_seeds=(1, 2), query_seeds=(2, 3))
    plan = QueryEvidencePlan(support_seeds=(1, 2), query_seeds=(101, 102))
    policy = FrozenPolicy()

    evidence = evaluate_frozen_policy(
        label='clean', policy=policy, plan=plan,
        query=lambda _, seed: (float(seed), np.array([seed, 0, 0], dtype=float)),
    )
    assert evidence.mean_objective == pytest.approx(101.5)
    assert evidence.query_seeds == (101, 102)

    def mutate(current, seed):
        current.version += 1
        return float(seed), np.zeros(3)

    with pytest.raises(RuntimeError, match='mutated'):
        evaluate_frozen_policy(label='bad', policy=policy, plan=plan, query=mutate)


def test_pair_query_freezes_both_policies_and_records_both_objectives() -> None:
    defender = FrozenPolicy()
    attacker = FrozenPolicy()
    plan = QueryEvidencePlan((1,), (10, 11))
    evidence = evaluate_frozen_pair(
        label='pair', defender=defender, attacker=attacker, plan=plan,
        query=lambda _, __, seed: (
            float(seed), float(-seed),
            np.array([0.1, 0.2, 0.3]), np.array([-0.1, -0.2, -0.3]),
        ),
    )
    assert evidence.mean_defender_objective == 10.5
    assert evidence.mean_attacker_objective == -10.5
    assert evidence.defender_fingerprint == evidence.attacker_fingerprint == '0'


def test_scientific_gate_checks_all_six_predeclared_comparisons() -> None:
    evidence = {
        'attacker_initial': _evidence('attacker_initial', 1.0, (0, 0, 0)),
        'attacker_br': _evidence('attacker_br', 1.3, (0.3, 0, 0)),
        'defender_a_response': _evidence('defender_a_response', 1.3, (0.3, 0, 0)),
        'defender_b_response': _evidence('defender_b_response', 1.0, (-0.3, 0, 0)),
        'defender_initial': _evidence('defender_initial', 0.4, (0, 0, 0)),
        'defender_adapted': _evidence('defender_adapted', 0.7, (0.3, 0, 0)),
        'meta_adapted': _evidence('meta_adapted', 0.8, (0.4, 0, 0)),
        'random_adapted': _evidence('random_adapted', 0.5, (0.1, 0, 0)),
        'no_adaptation': _evidence('no_adaptation', 0.4, (0, 0, 0)),
        'learned_defender': _evidence('learned_defender', 0.85, (0.4, 0, 0)),
        'specialized_oracle': _evidence('specialized_oracle', 0.9, (0.5, 0, 0)),
    }
    thresholds = ScientificGateThresholds(
        attacker_improvement=0.1,
        response_difference=0.1,
        defender_adaptation_improvement=0.1,
        meta_advantage=0.1,
        oracle_regret=0.1,
        action_difference=0.1,
    )

    result = evaluate_meta_sg_scientific_gate(evidence, thresholds)

    assert result.passed
    assert len(result.checks) == 6
    assert all(check.passed for check in result.checks)


def test_scientific_gate_preserves_failure_without_threshold_tuning() -> None:
    same = _evidence('same', 0.0, (0, 0, 0))
    labels = (
        'attacker_initial', 'attacker_br', 'defender_a_response',
        'defender_b_response', 'defender_initial', 'defender_adapted',
        'meta_adapted', 'random_adapted', 'no_adaptation',
        'learned_defender', 'specialized_oracle',
    )
    evidence = {label: QueryPolicyEvidence(
        label, same.query_seeds, same.mean_objective,
        same.action_trajectory, same.policy_fingerprint,
    ) for label in labels}
    evidence['specialized_oracle'] = _evidence('specialized_oracle', 1.0, (0, 0, 0))
    thresholds = ScientificGateThresholds(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

    result = evaluate_meta_sg_scientific_gate(evidence, thresholds)

    assert not result.passed
    assert {check.name for check in result.checks if not check.passed} == {
        'attacker_best_response', 'defender_conditioned_response',
        'defender_task_adaptation', 'meta_initialization',
        'specialized_oracle_regret', 'behavior_and_objective_signal',
    }
