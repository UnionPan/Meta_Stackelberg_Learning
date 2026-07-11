"""One-call deterministic scaled Meta-SG training and scientific evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.deterministic_paper_env import (
    make_deterministic_paper_env,
)
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
    ScaledPaperMetaSGTrainingRunner,
    ScaledPolicyTrainingResult,
)
from meta_stackelberg.experiments.scientific_gate import (
    QueryEvidencePlan,
    ScientificGateThresholds,
)
from meta_stackelberg.experiments.scientific_run import (
    PaperScientificGateRunner,
    PaperScientificRunResult,
)


@dataclass(frozen=True)
class ScaledEvidenceResult:
    training: ScaledPolicyTrainingResult
    scientific: PaperScientificRunResult
    parameter_snapshot: Mapping[str, object]
    query_seeds: tuple[int, ...]
    protocol: str = 'scaled-meta-sg-evidence-v1'


DeterministicScaledEvidenceResult = ScaledEvidenceResult


def run_deterministic_scaled_evidence(
    *,
    config: ScaledMetaSGConfig,
    thresholds: ScientificGateThresholds,
    query_seeds: tuple[int, ...],
    training_support_seed: int,
    scientific_support_seeds: tuple[int, ...],
    seed: int,
    attack_domain: AttackTypeDomainSource | None = None,
) -> ScaledEvidenceResult:
    """Run Algorithm 1/2 then all held-out comparisons without test helpers."""
    if not isinstance(config, ScaledMetaSGConfig):
        raise TypeError('config must be ScaledMetaSGConfig')
    probe = make_deterministic_paper_env(seed=seed, horizon=config.H)
    defender_obs_dim = len(flatten_observation(
        probe.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_probe = probe.observation_encoder.attacker_observation(
        probe.defender_observation(),
        malicious_count=0,
        defender_raw_action=np.zeros(3, dtype=np.float32),
    )
    attacker_obs_dim = len(flatten_observation(
        attacker_probe, ATTACKER_OBSERVATION_KEYS,
    ))

    def env_factory(task, rollout_seed, horizon):
        return make_deterministic_paper_env(
            seed=rollout_seed, horizon=horizon, task_id=str(task),
        )

    return run_scaled_evidence(
        config=config,
        thresholds=thresholds,
        query_seeds=query_seeds,
        training_support_seed=training_support_seed,
        scientific_support_seeds=scientific_support_seeds,
        seed=seed,
        env_factory=env_factory,
        defender_obs_dim=defender_obs_dim,
        attacker_obs_dim=attacker_obs_dim,
        attack_domain=attack_domain,
    )


def run_scaled_evidence(
    *,
    config: ScaledMetaSGConfig,
    thresholds: ScientificGateThresholds,
    query_seeds: tuple[int, ...],
    training_support_seed: int,
    scientific_support_seeds: tuple[int, ...],
    seed: int,
    env_factory,
    defender_obs_dim: int,
    attacker_obs_dim: int,
    attack_domain: AttackTypeDomainSource | None = None,
) -> ScaledEvidenceResult:
    """Shared Algorithm 1/2 + frozen-query protocol for any paper environment."""
    if not isinstance(config, ScaledMetaSGConfig):
        raise TypeError('config must be ScaledMetaSGConfig')
    initial_defender = _agent(
        config, defender_obs_dim, 'defender', seed + 1,
    )
    random_defender = _agent(
        config, defender_obs_dim, 'defender', seed + 2,
    )
    if attack_domain is None:
        tasks = tuple(f'rl-{index}' for index in range(config.K))
        initial_attackers = {
            task: _agent(config, attacker_obs_dim, 'attacker', seed + 10 + index)
            for index, task in enumerate(tasks)
        }
        attack_origins = {task: 'random-untrained' for task in tasks}
        attack_protocol = 'random-untrained-explicit-v1'
    else:
        if len(attack_domain.snapshots) != config.K:
            raise ValueError('attack domain size must equal K')
        tasks = tuple(attack_domain.snapshots)
        initial_attackers = {}
        for index, task in enumerate(tasks):
            policy = _agent(config, attacker_obs_dim, 'attacker', seed + 10 + index)
            policy.restore(attack_domain.snapshots[task])
            initial_attackers[task] = policy
        attack_origins = dict(attack_domain.origins)
        attack_protocol = attack_domain.protocol

    training = ScaledPaperMetaSGTrainingRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_obs_dim,
        attacker_obs_dim=attacker_obs_dim,
        support_seed=training_support_seed,
        query_seeds=query_seeds,
    ).run(
        initial_defender=initial_defender,
        initial_attackers=initial_attackers,
        sample_tasks=lambda iteration, count: tasks,
    )
    first_attacker = initial_attackers[tasks[0]]
    scientific = PaperScientificGateRunner(
        config=config,
        env_factory=env_factory,
        defender_obs_dim=defender_obs_dim,
        attacker_obs_dim=attacker_obs_dim,
        evidence_plan=QueryEvidencePlan(
            scientific_support_seeds, query_seeds,
        ),
        thresholds=thresholds,
    ).run(
        task=tasks[0],
        learned_defender=training.algorithm1_defender,
        random_defender=random_defender,
        initial_attacker=first_attacker,
        specialized_defenders={
            'low-alpha': _constant_policy(
                initial_defender, (-0.5, 0.0, 0.0),
            ),
            'high-alpha': _constant_policy(
                initial_defender, (0.5, 0.0, 0.0),
            ),
        },
        attacker_oracle_policies={
            'low-gamma': _constant_policy(
                first_attacker, (-0.8, 0.0, 0.0),
            ),
            'high-gamma': _constant_policy(
                first_attacker, (0.8, 0.0, 0.0),
            ),
        },
    )
    parameters = MappingProxyType({
        'T': config.T,
        'K': config.K,
        'H': config.H,
        'l': config.l,
        'N_A': config.N_A,
        'N_D': config.N_D,
        'workers': config.workers,
        'untargeted_attackers': config.untargeted_attackers,
        'sample_size': config.sample_size,
        'td3_batch_size': config.td3_batch_size,
        'learning_starts': config.learning_starts,
        'hidden_sizes': config.hidden_sizes,
        'replay_capacity': config.replay_capacity,
        'scale_provenance': config.scale_provenance,
        'attack_domain_protocol': attack_protocol,
        'attack_type_origins': attack_origins,
    })
    return ScaledEvidenceResult(
        training, scientific, parameters, query_seeds,
    )


def _agent(config, obs_dim, role, seed):
    paper = config.paper_reference
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role=role,
        seed=seed,
        hidden_sizes=config.hidden_sizes,
        learning_rate=paper.policy_learning_rate,
        gamma=paper.gamma,
        tau=paper.tau,
        policy_delay=paper.policy_delay,
        target_policy_noise=paper.target_policy_noise,
        noise_clip=paper.noise_clip,
    )


def _constant_policy(source, raw_action):
    result = source.clone()
    action = torch.as_tensor(raw_action, dtype=torch.float32)
    if action.shape != (3,) or torch.any(action <= -1) or torch.any(action >= 1):
        raise ValueError('constant raw action must have shape (3,) within (-1,1)')
    with torch.no_grad():
        for actor in (result.actor, result.actor_target):
            for parameter in actor.parameters():
                parameter.zero_()
            actor.model[-1].bias.copy_(torch.atanh(action))
    return result
