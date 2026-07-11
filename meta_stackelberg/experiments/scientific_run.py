"""End-to-end held-out scientific comparison protocol for Meta-SG policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.experiments.paper_meta_sg import PaperTD3TrajectoryCollector
from meta_stackelberg.experiments.scientific_gate import (
    MetaSGScientificGateResult,
    QueryEvidencePlan,
    QueryPairEvidence,
    QueryPolicyEvidence,
    ScientificGateThresholds,
    evaluate_frozen_pair,
    evaluate_meta_sg_scientific_gate,
)
from meta_stackelberg.stackelberg.policy_adaptation import PolicyDefenderAdapter
from meta_stackelberg.stackelberg.policy_response import PolicyBestResponseTrainer


@dataclass(frozen=True)
class ScientificTrainingBudget:
    trajectories: int
    fl_rounds: int
    td3_updates: int


@dataclass(frozen=True)
class PaperScientificRunResult:
    gate: MetaSGScientificGateResult
    evidence: Mapping[str, QueryPolicyEvidence]
    pair_evidence: Mapping[str, QueryPairEvidence]
    budgets: Mapping[str, ScientificTrainingBudget]
    used_support_seeds: tuple[int, ...]
    query_seeds: tuple[int, ...]
    fresh_response_count: int
    specialized_oracle_label: str
    adaptation_seed_blocks: Mapping[str, tuple[int, ...]]
    protocol: str = 'paper-meta-sg-held-out-comparisons-v1'


class PaperScientificGateRunner:
    """Train fresh policy BRs and execute all six held-out comparisons."""

    def __init__(
        self,
        *,
        config: ScaledMetaSGConfig,
        env_factory,
        defender_obs_dim: int,
        attacker_obs_dim: int,
        evidence_plan: QueryEvidencePlan,
        thresholds: ScientificGateThresholds,
    ) -> None:
        if not isinstance(config, ScaledMetaSGConfig):
            raise TypeError('config must be ScaledMetaSGConfig')
        self.config = config
        self.env_factory = env_factory
        self.defender_obs_dim = defender_obs_dim
        self.attacker_obs_dim = attacker_obs_dim
        self.plan = evidence_plan
        self.thresholds = thresholds
        self._support_cursor = 0
        self._used_support_seeds = []
        self._replay_serial = 0
        self._fresh_response_count = 0
        self._seed_override = None
        required = max(config.td3_batch_size, config.learning_starts)
        self.trajectories_per_update = max(1, int(np.ceil(required / config.H)))

    def run(
        self,
        *,
        task,
        learned_defender: TD3Agent,
        meta_defender: TD3Agent,
        random_defender: TD3Agent,
        initial_attacker: TD3Agent,
        specialized_defenders: Mapping[str, TD3Agent],
    ) -> PaperScientificRunResult:
        if not specialized_defenders:
            raise ValueError('predeclared specialized defender grid must not be empty')
        pair_evidence = {}
        budgets = {}

        learned_br = self._fresh_br(task, learned_defender, initial_attacker)
        pair_evidence['attacker_initial'] = self._query_pair(
            'attacker_initial', task, learned_defender, initial_attacker,
        )
        pair_evidence['attacker_br'] = self._query_pair(
            'attacker_br', task, learned_defender, learned_br,
        )
        pair_evidence['defender_a_response'] = pair_evidence['attacker_br']

        random_br = self._fresh_br(task, random_defender, initial_attacker)
        pair_evidence['defender_b_response'] = self._query_pair(
            'defender_b_response', task, random_defender, random_br,
        )
        pair_evidence['defender_initial'] = pair_evidence['attacker_br']

        learned_adapted, budgets['defender_adapted'] = self._adapt(
            task, learned_defender, learned_br,
        )
        learned_adapted_br = self._fresh_br(
            task, learned_adapted, initial_attacker,
        )
        pair_evidence['defender_adapted'] = self._query_pair(
            'defender_adapted', task, learned_adapted, learned_adapted_br,
        )
        budgets['defender_initial'] = ScientificTrainingBudget(0, 0, 0)

        meta_br = self._fresh_br(task, meta_defender, initial_attacker)
        meta_seed_start = len(self._used_support_seeds)
        meta_adapted, budgets['meta_adapted'] = self._adapt(
            task, meta_defender, meta_br,
        )
        matched_adaptation_seeds = tuple(
            self._used_support_seeds[meta_seed_start:]
        )
        meta_adapted_br = self._fresh_br(
            task, meta_adapted, initial_attacker,
        )
        pair_evidence['meta_adapted'] = self._query_pair(
            'meta_adapted', task, meta_adapted, meta_adapted_br,
        )

        random_adapted, budgets['random_adapted'] = self._adapt(
            task, random_defender, random_br,
            seed_override=matched_adaptation_seeds,
        )
        random_adapted_br = self._fresh_br(
            task, random_adapted, initial_attacker,
        )
        pair_evidence['random_adapted'] = self._query_pair(
            'random_adapted', task, random_adapted, random_adapted_br,
        )

        no_adaptation, budgets['no_adaptation'] = self._consume_no_adaptation_budget(
            task, meta_defender, meta_br,
            seed_override=matched_adaptation_seeds,
        )
        pair_evidence['no_adaptation'] = self._query_pair(
            'no_adaptation', task, no_adaptation, meta_br,
        )
        pair_evidence['learned_defender'] = pair_evidence['defender_adapted']

        specialized_pairs = {}
        for label, specialized in specialized_defenders.items():
            response = self._fresh_br(task, specialized, initial_attacker)
            specialized_pairs[label] = self._query_pair(
                f'specialized:{label}', task, specialized, response,
            )
        oracle_label, oracle_pair = max(
            specialized_pairs.items(),
            key=lambda item: item[1].mean_defender_objective,
        )
        pair_evidence['specialized_oracle'] = oracle_pair

        evidence = {
            'attacker_initial': _attacker_evidence(
                'attacker_initial', pair_evidence['attacker_initial'],
            ),
            'attacker_br': _attacker_evidence(
                'attacker_br', pair_evidence['attacker_br'],
            ),
            'defender_a_response': _attacker_evidence(
                'defender_a_response', pair_evidence['defender_a_response'],
            ),
            'defender_b_response': _attacker_evidence(
                'defender_b_response', pair_evidence['defender_b_response'],
            ),
        }
        for label in (
            'defender_initial', 'defender_adapted', 'meta_adapted',
            'random_adapted', 'no_adaptation', 'learned_defender',
            'specialized_oracle',
        ):
            evidence[label] = _defender_evidence(label, pair_evidence[label])
        gate = evaluate_meta_sg_scientific_gate(evidence, self.thresholds)
        return PaperScientificRunResult(
            gate,
            evidence,
            pair_evidence,
            budgets,
            tuple(self._used_support_seeds),
            self.plan.query_seeds,
            self._fresh_response_count,
            oracle_label,
            {
                'meta_adapted': matched_adaptation_seeds,
                'random_adapted': matched_adaptation_seeds,
                'no_adaptation': matched_adaptation_seeds,
            },
        )

    def _fresh_br(self, task, defender, initial_attacker) -> TD3Agent:
        attacker = initial_attacker.clone()
        replay = self._new_replay('attacker')
        PolicyBestResponseTrainer(
            N_A=self.config.N_A,
            batch_size=self.config.td3_batch_size,
            kappa_A=self.config.paper_reference.kappa_attacker,
        ).train(
            defender=defender,
            attacker=attacker,
            replay=replay,
            collect_fresh=lambda step: self._collect_updates(
                task, defender, attacker, replay, 'attacker', step,
            ),
            independent_objective=lambda policy: float(np.mean(policy.act(
                np.zeros(self.attacker_obs_dim, dtype=np.float32),
                deterministic=True,
            ))),
        )
        self._fresh_response_count += 1
        return attacker

    def _adapt(self, task, defender, attacker, *, seed_override=None):
        replay = self._new_replay('defender')
        self._begin_seed_override(seed_override)
        try:
            result = PolicyDefenderAdapter(
                l=self.config.l,
                batch_size=self.config.td3_batch_size,
                eta=self.config.paper_reference.adaptation_step,
            ).adapt(
                defender=defender,
                attacker=attacker,
                replay=replay,
                collect_fresh=lambda adapted, frozen, target: self._collect_updates(
                    task, adapted, frozen, target, 'defender', 0,
                ),
            )
        finally:
            self._end_seed_override()
        return result.adapted_defender, self._adaptation_budget(self.config.l)

    def _consume_no_adaptation_budget(
        self, task, defender, attacker, *, seed_override=None,
    ):
        result = defender.clone()
        replay = self._new_replay('defender')
        self._begin_seed_override(seed_override)
        try:
            for step in range(self.config.l):
                self._collect_updates(
                    task, result, attacker, replay, 'defender', step,
                )
        finally:
            self._end_seed_override()
        budget = self._adaptation_budget(self.config.l, td3_updates=0)
        return result, budget

    def _adaptation_budget(self, updates, *, td3_updates=None):
        trajectories = updates * self.trajectories_per_update
        return ScientificTrainingBudget(
            trajectories,
            trajectories * self.config.H,
            updates if td3_updates is None else td3_updates,
        )

    def _query_pair(self, label, task, defender, attacker):
        return evaluate_frozen_pair(
            label=label,
            defender=defender,
            attacker=attacker,
            plan=self.plan,
            query=lambda current_defender, current_attacker, seed: self._query_trajectory(
                task, current_defender, current_attacker, seed,
            ),
        )

    def _query_trajectory(self, task, defender, attacker, seed):
        trajectory = PaperTD3TrajectoryCollector().collect(
            env=self.env_factory(task, seed, self.config.H),
            defender=defender,
            attacker=attacker,
            defender_replay=self._new_replay('defender'),
            attacker_replay=self._new_replay('attacker'),
            generation=0,
            deterministic=True,
        )
        return (
            trajectory.defender_return,
            trajectory.attacker_return,
            np.concatenate([step.defender_raw_action for step in trajectory.steps]),
            np.concatenate([step.attacker_raw_action for step in trajectory.steps]),
        )

    def _collect_updates(
        self, task, defender, attacker, target_replay, target_role, generation,
    ):
        for _ in range(self.trajectories_per_update):
            seed = self._next_support_seed()
            trajectory = PaperTD3TrajectoryCollector().collect(
                env=self.env_factory(task, seed, self.config.H),
                defender=defender,
                attacker=attacker,
                defender_replay=(
                    target_replay if target_role == 'defender'
                    else self._new_replay('defender')
                ),
                attacker_replay=(
                    target_replay if target_role == 'attacker'
                    else self._new_replay('attacker')
                ),
                generation=generation,
                deterministic=False,
                explore_role=target_role,
            )
            del trajectory

    def _next_support_seed(self):
        if self._seed_override is not None:
            try:
                seed = next(self._seed_override)
            except StopIteration as error:
                raise RuntimeError('matched support seed block exhausted') from error
            self._used_support_seeds.append(seed)
            return seed
        if self._support_cursor >= len(self.plan.support_seeds):
            raise RuntimeError('predeclared support seed budget exhausted')
        seed = self.plan.support_seeds[self._support_cursor]
        self._support_cursor += 1
        self._used_support_seeds.append(seed)
        return seed

    def _begin_seed_override(self, seeds):
        if seeds is None:
            return
        if self._seed_override is not None:
            raise RuntimeError('nested support seed override is not allowed')
        self._seed_override = iter(seeds)

    def _end_seed_override(self):
        self._seed_override = None

    def _new_replay(self, role):
        self._replay_serial += 1
        return TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=(self.defender_obs_dim if role == 'defender' else self.attacker_obs_dim),
            action_dim=3,
            role=role,
            seed=20_000_000 + self._replay_serial,
        )


def _attacker_evidence(label, pair):
    return QueryPolicyEvidence(
        label,
        pair.query_seeds,
        pair.mean_attacker_objective,
        pair.attacker_action_trajectories,
        pair.attacker_fingerprint,
    )


def _defender_evidence(label, pair):
    return QueryPolicyEvidence(
        label,
        pair.query_seeds,
        pair.mean_defender_objective,
        pair.defender_action_trajectories,
        pair.defender_fingerprint,
    )
