"""Matched-budget scientific evidence for MNIST white-box Meta-SG."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.experiments.paper_meta_sg import PaperTD3TrajectoryCollector
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    ScientificGateThresholds,
)
from meta_stackelberg.experiments.scientific_run import (
    PaperScientificGateRunner,
    PaperScientificRunResult,
)
from meta_stackelberg.experiments.whitebox_backdoor_evidence import (
    WhiteBoxBackdoorMetricEvidence,
    WhiteBoxBackdoorMetrics,
    WhiteBoxSafetyGateResult,
    WhiteBoxSafetyThresholds,
    evaluate_whitebox_backdoor_model,
    evaluate_whitebox_safety_gate,
)
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger


@dataclass(frozen=True)
class MNISTWhiteBoxScientificResult:
    meta_sg: PaperScientificRunResult
    metrics: Mapping[str, WhiteBoxBackdoorMetricEvidence]
    safety_gate: WhiteBoxSafetyGateResult
    passed: bool
    protocol: str = 'mnist-whitebox-scientific-evidence-v1'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metrics', MappingProxyType(dict(self.metrics)))
        if self.passed != (self.meta_sg.gate.passed and self.safety_gate.passed):
            raise ValueError('combined white-box pass must require both gates')


class _MetricScientificRunner(PaperScientificGateRunner):
    def __init__(
        self,
        *,
        environment_factory: PaperMNISTBackdoorEnvironmentFactory,
        query_batch_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if query_batch_size <= 0:
            raise ValueError('query_batch_size must be positive')
        self.dataset_factory = environment_factory
        self.query_batch_size = int(query_batch_size)
        self.metric_records: dict[str, list[WhiteBoxBackdoorMetrics]] = {}
        self._active_query_label: str | None = None

    def _query_pair(self, label, task, defender, attacker):
        if self._active_query_label is not None:
            raise RuntimeError('nested white-box query labels are not allowed')
        self._active_query_label = label
        self.metric_records[label] = []
        try:
            result = super()._query_pair(label, task, defender, attacker)
        finally:
            self._active_query_label = None
        if len(self.metric_records[label]) != len(self.plan.query_seeds):
            raise RuntimeError('white-box query metric count does not match seeds')
        return result

    def _query_trajectory(self, task, defender, attacker, seed):
        if self._active_query_label is None:
            raise RuntimeError('white-box query trajectory has no evidence label')
        env = self.env_factory(task, seed, self.config.H)
        trajectory = PaperTD3TrajectoryCollector().collect(
            env=env,
            defender=defender,
            attacker=attacker,
            defender_replay=self._new_replay('defender'),
            attacker_replay=self._new_replay('attacker'),
            generation=0,
            deterministic=True,
        )
        delivered = env.final_delivered_model()
        if delivered is None:
            raise RuntimeError('query trajectory did not produce a delivered model')
        fixture = mnist_global_trigger()
        self.metric_records[self._active_query_label].append(
            evaluate_whitebox_backdoor_model(
                model=delivered,
                query_dataset=self.dataset_factory.datasets.query,
                trigger=fixture.trigger,
                source_class=fixture.source_class,
                target_class=fixture.target_class,
                batch_size=self.query_batch_size,
            )
        )
        return (
            trajectory.mean_defender_reward,
            trajectory.mean_attacker_reward,
            np.concatenate([
                step.defender_raw_action for step in trajectory.steps
            ]),
            np.concatenate([
                step.attacker_raw_action for step in trajectory.steps
            ]),
        )


def run_mnist_whitebox_scientific_evidence(
    *,
    config: ScaledMetaSGConfig,
    environment_factory: PaperMNISTBackdoorEnvironmentFactory,
    evidence_plan: QueryEvidencePlan,
    meta_thresholds: ScientificGateThresholds,
    safety_thresholds: WhiteBoxSafetyThresholds,
    learned_defender: TD3Agent,
    random_defender: TD3Agent,
    initial_attacker: TD3Agent,
    specialized_defenders: Mapping[str, TD3Agent],
    attacker_oracle_policies: Mapping[str, TD3Agent],
    task: str,
    query_batch_size: int = 128,
) -> MNISTWhiteBoxScientificResult:
    if not isinstance(environment_factory, PaperMNISTBackdoorEnvironmentFactory):
        raise TypeError('environment_factory must be PaperMNISTBackdoorEnvironmentFactory')

    def env_factory(current_task, seed, horizon):
        return environment_factory.make(
            seed=seed,
            horizon=horizon,
            task_id=f'mnist-whitebox-scientific:{current_task}',
        )

    runner = _MetricScientificRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=environment_factory.defender_observation_dim,
        attacker_obs_dim=environment_factory.attacker_observation_dim,
        evidence_plan=evidence_plan,
        thresholds=meta_thresholds,
        environment_factory=environment_factory,
        query_batch_size=query_batch_size,
    )
    meta_result = runner.run(
        task=task,
        learned_defender=learned_defender,
        random_defender=random_defender,
        initial_attacker=initial_attacker,
        specialized_defenders=specialized_defenders,
        attacker_oracle_policies=attacker_oracle_policies,
    )
    source_labels = {
        'attacker_initial': 'attacker_initial',
        'attacker_br': 'attacker_br',
        'attacker_oracle': f'attacker-oracle:{meta_result.attacker_oracle_label}',
        'defender_a_response': 'attacker_br',
        'defender_b_response': 'defender_b_response',
        'defender_initial': 'attacker_br',
        'defender_adapted': 'defender_adapted',
        'meta_adapted': 'defender_adapted',
        'random_adapted': 'random_adapted',
        'no_adaptation': 'no_adaptation',
        'learned_defender': 'defender_adapted',
        'specialized_oracle': f'specialized:{meta_result.specialized_oracle_label}',
    }
    metrics = {
        label: WhiteBoxBackdoorMetricEvidence(
            label=label,
            query_seeds=evidence_plan.query_seeds,
            per_seed=tuple(runner.metric_records[source]),
        )
        for label, source in source_labels.items()
    }
    safety_gate = evaluate_whitebox_safety_gate(metrics, safety_thresholds)
    return MNISTWhiteBoxScientificResult(
        meta_sg=meta_result,
        metrics=metrics,
        safety_gate=safety_gate,
        passed=meta_result.gate.passed and safety_gate.passed,
    )
