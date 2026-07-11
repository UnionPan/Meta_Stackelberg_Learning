"""Predeclared, frozen-query scientific gates for paper-aligned Meta-SG."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class QueryEvidencePlan:
    support_seeds: tuple[int, ...]
    query_seeds: tuple[int, ...]
    protocol: str = 'held-out-frozen-query-v1'

    def __post_init__(self) -> None:
        for seeds, name in (
            (self.support_seeds, 'support_seeds'),
            (self.query_seeds, 'query_seeds'),
        ):
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError(f'{name} must be non-empty and unique')
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
                raise TypeError(f'{name} must contain integers')
        if set(self.support_seeds) & set(self.query_seeds):
            raise ValueError('support and query seeds must be disjoint')


@dataclass(frozen=True)
class QueryPolicyEvidence:
    label: str
    query_seeds: tuple[int, ...]
    mean_objective: float
    action_trajectory: tuple[tuple[float, ...], ...]
    policy_fingerprint: str

    def __post_init__(self) -> None:
        if not self.label or not self.policy_fingerprint:
            raise ValueError('evidence label and fingerprint must not be empty')
        if not self.query_seeds or len(self.query_seeds) != len(self.action_trajectory):
            raise ValueError('one action trajectory record is required per query seed')
        if not math.isfinite(self.mean_objective):
            raise ValueError('mean objective must be finite')
        if any(not action or not all(math.isfinite(value) for value in action)
               for action in self.action_trajectory):
            raise ValueError('query actions must be non-empty and finite')


@dataclass(frozen=True)
class QueryPairEvidence:
    label: str
    query_seeds: tuple[int, ...]
    mean_defender_objective: float
    mean_attacker_objective: float
    defender_action_trajectories: tuple[tuple[float, ...], ...]
    attacker_action_trajectories: tuple[tuple[float, ...], ...]
    defender_fingerprint: str
    attacker_fingerprint: str


@dataclass(frozen=True)
class ScientificGateThresholds:
    attacker_improvement: float
    response_difference: float
    defender_adaptation_improvement: float
    meta_advantage: float
    oracle_regret: float
    action_difference: float
    attacker_plateau_gap: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')


@dataclass(frozen=True)
class ScientificGateCheck:
    name: str
    passed: bool
    observed: float
    threshold: float
    comparison: str
    alternative_observed: float | None = None
    alternative_threshold: float | None = None


@dataclass(frozen=True)
class MetaSGScientificGateResult:
    passed: bool
    checks: tuple[ScientificGateCheck, ...]
    thresholds: ScientificGateThresholds
    query_seeds: tuple[int, ...]
    protocol: str = 'meta-sg-scientific-gate-v1'


def evaluate_frozen_policy(
    *,
    label: str,
    policy,
    plan: QueryEvidencePlan,
    query: Callable[[object, int], tuple[float, np.ndarray]],
) -> QueryPolicyEvidence:
    """Evaluate query seeds without exposing an update path to this function."""
    before = policy.fingerprint()
    objectives = []
    actions = []
    for seed in plan.query_seeds:
        objective, action = query(policy, seed)
        objective = float(objective)
        flat_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if not math.isfinite(objective) or flat_action.size == 0 or not np.all(np.isfinite(flat_action)):
            raise ValueError('query returned non-finite objective or action')
        objectives.append(objective)
        actions.append(tuple(float(value) for value in flat_action))
        if policy.fingerprint() != before:
            raise RuntimeError('frozen policy mutated during query evaluation')
    return QueryPolicyEvidence(
        label,
        plan.query_seeds,
        float(np.mean(objectives)),
        tuple(actions),
        before,
    )


def evaluate_frozen_pair(
    *,
    label: str,
    defender,
    attacker,
    plan: QueryEvidencePlan,
    query,
) -> QueryPairEvidence:
    """Run held-out pair trajectories while freezing both complete policies."""
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()
    defender_objectives = []
    attacker_objectives = []
    defender_actions = []
    attacker_actions = []
    for seed in plan.query_seeds:
        defender_objective, attacker_objective, defender_action, attacker_action = query(
            defender, attacker, seed,
        )
        defender_array = np.asarray(defender_action, dtype=np.float64).reshape(-1)
        attacker_array = np.asarray(attacker_action, dtype=np.float64).reshape(-1)
        values = (float(defender_objective), float(attacker_objective))
        if (
            not all(math.isfinite(value) for value in values)
            or defender_array.size == 0
            or attacker_array.size == 0
            or not np.all(np.isfinite(defender_array))
            or not np.all(np.isfinite(attacker_array))
        ):
            raise ValueError('pair query returned non-finite objective or action')
        defender_objectives.append(values[0])
        attacker_objectives.append(values[1])
        defender_actions.append(tuple(float(value) for value in defender_array))
        attacker_actions.append(tuple(float(value) for value in attacker_array))
        if defender.fingerprint() != defender_before:
            raise RuntimeError('frozen defender mutated during pair query')
        if attacker.fingerprint() != attacker_before:
            raise RuntimeError('frozen attacker mutated during pair query')
    return QueryPairEvidence(
        label,
        plan.query_seeds,
        float(np.mean(defender_objectives)),
        float(np.mean(attacker_objectives)),
        tuple(defender_actions),
        tuple(attacker_actions),
        defender_before,
        attacker_before,
    )


_REQUIRED_LABELS = (
    'attacker_initial', 'attacker_br',
    'defender_a_response', 'defender_b_response',
    'defender_initial', 'defender_adapted',
    'meta_adapted', 'random_adapted', 'no_adaptation',
    'learned_defender', 'specialized_oracle',
)


def evaluate_meta_sg_scientific_gate(
    evidence: Mapping[str, QueryPolicyEvidence],
    thresholds: ScientificGateThresholds,
) -> MetaSGScientificGateResult:
    missing = set(_REQUIRED_LABELS) - set(evidence)
    if missing:
        raise ValueError(f'missing scientific evidence labels: {sorted(missing)}')
    query_seeds = evidence[_REQUIRED_LABELS[0]].query_seeds
    for label in _REQUIRED_LABELS:
        item = evidence[label]
        if item.label != label or item.query_seeds != query_seeds:
            raise ValueError('all evidence must use matching labels and query seeds')

    attacker_gain = _objective(evidence, 'attacker_br') - _objective(evidence, 'attacker_initial')
    plateau_gap = math.inf
    if 'attacker_oracle' in evidence:
        oracle = evidence['attacker_oracle']
        if oracle.query_seeds != query_seeds:
            raise ValueError('attacker oracle must use the same query seeds')
        plateau_gap = oracle.mean_objective - _objective(evidence, 'attacker_br')
    attacker_pass = (
        attacker_gain >= thresholds.attacker_improvement
        or plateau_gap <= thresholds.attacker_plateau_gap
    )
    response_difference = max(
        abs(_objective(evidence, 'defender_a_response') - _objective(evidence, 'defender_b_response')),
        _action_distance(evidence['defender_a_response'], evidence['defender_b_response']),
    )
    adaptation_gain = _objective(evidence, 'defender_adapted') - _objective(evidence, 'defender_initial')
    meta_margin = min(
        _objective(evidence, 'meta_adapted') - _objective(evidence, 'random_adapted'),
        _objective(evidence, 'meta_adapted') - _objective(evidence, 'no_adaptation'),
    )
    oracle_regret = _objective(evidence, 'specialized_oracle') - _objective(evidence, 'learned_defender')
    action_signal = min(
        _action_distance(evidence['attacker_br'], evidence['attacker_initial']),
        _action_distance(evidence['defender_adapted'], evidence['defender_initial']),
    )
    objective_signal = min(attacker_gain, adaptation_gain, meta_margin)
    checks = (
        ScientificGateCheck('attacker_best_response', attacker_pass, attacker_gain,
                            thresholds.attacker_improvement, 'phi(N_A) - phi(0), or oracle plateau',
                            plateau_gap if math.isfinite(plateau_gap) else None,
                            thresholds.attacker_plateau_gap),
        ScientificGateCheck('defender_conditioned_response', response_difference >= thresholds.response_difference,
                            response_difference, thresholds.response_difference, 'response A vs response B'),
        ScientificGateCheck('defender_task_adaptation', adaptation_gain >= thresholds.defender_adaptation_improvement,
                            adaptation_gain, thresholds.defender_adaptation_improvement, 'adapted - initial'),
        ScientificGateCheck('meta_initialization', meta_margin >= thresholds.meta_advantage,
                            meta_margin, thresholds.meta_advantage, 'meta vs random and no-adaptation'),
        ScientificGateCheck('specialized_oracle_regret', oracle_regret <= thresholds.oracle_regret,
                            oracle_regret, thresholds.oracle_regret, 'oracle - learned'),
        ScientificGateCheck('behavior_and_objective_signal',
                            action_signal >= thresholds.action_difference and objective_signal > 0,
                            action_signal, thresholds.action_difference,
                            'held-out action and objective signals',
                            objective_signal, 0.0),
    )
    return MetaSGScientificGateResult(
        all(check.passed for check in checks), checks, thresholds, query_seeds,
    )


def _objective(evidence: Mapping[str, QueryPolicyEvidence], label: str) -> float:
    return evidence[label].mean_objective


def _action_distance(left: QueryPolicyEvidence, right: QueryPolicyEvidence) -> float:
    left_array = np.concatenate([np.asarray(action) for action in left.action_trajectory])
    right_array = np.concatenate([np.asarray(action) for action in right.action_trajectory])
    if left_array.shape != right_array.shape:
        raise ValueError('compared action trajectories must have equal shapes')
    return float(np.linalg.norm(left_array - right_array) / math.sqrt(left_array.size))
