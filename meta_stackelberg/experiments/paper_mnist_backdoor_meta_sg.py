"""Algorithm 1 runner for the MNIST white-box backdoor game."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.experiments.attack_domain import (
    AttackTypeDomainSource,
    UniformAttackTypeSampler,
)
from meta_stackelberg.experiments.paper_meta_sg import PaperTD3TrajectoryCollector
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
)
from meta_stackelberg.stackelberg.policy_algorithm1 import (
    PolicyAlgorithm1Result,
    PolicyMetaSGAlgorithm1,
)


@dataclass(frozen=True)
class MNISTWhiteBoxMetaSGConfig:
    """Paper defaults with explicit smoke-scale overrides."""

    N_D: int = 10
    K: int = 10
    N_A: int = 10
    H: int = 200
    td3_batch_size: int = 256
    learning_starts: int = 100
    replay_capacity: int = 1_000_000
    hidden_sizes: tuple[int, ...] = (256, 256)
    policy_learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    eta: float = 0.01
    kappa_attacker: float = 0.001
    kappa_defender: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            'N_D', 'K', 'N_A', 'H', 'td3_batch_size', 'learning_starts',
            'replay_capacity', 'policy_delay',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.replay_capacity < self.td3_batch_size:
            raise ValueError('replay_capacity must cover td3_batch_size')
        if not self.hidden_sizes or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.hidden_sizes
        ):
            raise ValueError('hidden_sizes must contain positive integers')
        for name in (
            'policy_learning_rate', 'tau', 'target_policy_noise', 'noise_clip',
            'eta', 'kappa_attacker', 'kappa_defender',
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.gamma) or self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError('gamma must be within (0, 1]')


@dataclass(frozen=True)
class MNISTWhiteBoxMetaSGResult:
    algorithm1: PolicyAlgorithm1Result
    defender: TD3Agent
    attackers: Mapping[str, TD3Agent]
    support_seeds: tuple[int, ...]
    trajectory_count: int
    trajectories_per_update: int
    defender_action_dim: int = 3
    attacker_action_dim: int = 3
    protocol: str = 'mnist-whitebox-real-data-v1'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'attackers', MappingProxyType(dict(self.attackers)))


class _Algorithm1WhiteBoxRunner:
    def __init__(
        self,
        *,
        environment_factory: PaperMNISTBackdoorEnvironmentFactory,
        config: MNISTWhiteBoxMetaSGConfig,
        support_seed: int,
        query_seeds: tuple[int, ...],
    ) -> None:
        if support_seed < 0:
            raise ValueError('support_seed must be non-negative')
        self.environment_factory = environment_factory
        self.config = config
        self.next_support_seed = support_seed
        self.query_seeds = frozenset(query_seeds)
        self.support_seeds: list[int] = []
        self.replay_serial = 0
        required = max(config.td3_batch_size, config.learning_starts)
        self.trajectories_per_update = max(1, math.ceil(required / config.H))

    def run(
        self,
        *,
        defender: TD3Agent,
        attackers: Mapping[str, TD3Agent],
        sample_tasks,
    ) -> PolicyAlgorithm1Result:
        return PolicyMetaSGAlgorithm1(
            N_D=self.config.N_D,
            K=self.config.K,
            N_A=self.config.N_A,
            batch_size=self.config.td3_batch_size,
            eta=self.config.eta,
            kappa_A=self.config.kappa_attacker,
            kappa_D=self.config.kappa_defender,
        ).run(
            defender=defender,
            attackers=attackers,
            sample_tasks=sample_tasks,
            replay_factory=self._replay_factory,
            collect_adaptation=lambda task, adapted, frozen, replay, iteration: self._collect(
                task, adapted, frozen, replay, 'defender', iteration,
            ),
            collect_response=lambda task, step, frozen, attacker, replay, iteration: self._collect(
                task, frozen, attacker, replay, 'attacker', iteration,
            ),
            collect_leader=lambda task, adapted, frozen, replay, iteration: self._collect(
                task, adapted, frozen, replay, 'defender', iteration,
            ),
            independent_attacker_objective=lambda task, policy: float(np.mean(
                policy.act(
                    np.zeros(
                        self.environment_factory.attacker_observation_dim,
                        dtype=np.float32,
                    ),
                    deterministic=True,
                )
            )),
        )

    def _replay_factory(self, task, role, phase, iteration) -> TD3ReplayBuffer:
        del task, phase, iteration
        return self._new_replay(role)

    def _new_replay(self, role: str) -> TD3ReplayBuffer:
        self.replay_serial += 1
        dimension = (
            self.environment_factory.defender_observation_dim
            if role == 'defender'
            else self.environment_factory.attacker_observation_dim
        )
        return TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=dimension,
            action_dim=3,
            role=role,
            seed=20_000_000 + self.replay_serial,
        )

    def _collect(
        self,
        task: str,
        defender: TD3Agent,
        attacker: TD3Agent,
        target_replay: TD3ReplayBuffer,
        target_role: str,
        iteration: int,
    ) -> None:
        for _ in range(self.trajectories_per_update):
            seed = self.next_support_seed
            self.next_support_seed += 1
            if seed in self.query_seeds:
                raise RuntimeError('support trajectory seed overlaps held-out query seed')
            self.support_seeds.append(seed)
            env = self.environment_factory.make(
                seed=seed,
                horizon=self.config.H,
                task_id=f'mnist-whitebox-real-data-v1:{task}',
            )
            defender_replay = (
                target_replay
                if target_role == 'defender'
                else self._new_replay('defender')
            )
            attacker_replay = (
                target_replay
                if target_role == 'attacker'
                else self._new_replay('attacker')
            )
            PaperTD3TrajectoryCollector().collect(
                env=env,
                defender=defender,
                attacker=attacker,
                defender_replay=defender_replay,
                attacker_replay=attacker_replay,
                generation=iteration,
                deterministic=False,
                explore_role=target_role,
            )


def run_mnist_whitebox_backdoor_meta_sg(
    *,
    environment_factory: PaperMNISTBackdoorEnvironmentFactory,
    attack_domain: AttackTypeDomainSource,
    output_dir: str | Path,
    seed: int,
    support_seed: int,
    config: MNISTWhiteBoxMetaSGConfig | None = None,
    query_seeds: tuple[int, ...] = (),
) -> MNISTWhiteBoxMetaSGResult:
    if not isinstance(environment_factory, PaperMNISTBackdoorEnvironmentFactory):
        raise TypeError('environment_factory must be PaperMNISTBackdoorEnvironmentFactory')
    if not isinstance(attack_domain, AttackTypeDomainSource):
        raise TypeError('attack_domain must be AttackTypeDomainSource')
    origins = set(attack_domain.origins.values())
    if not {
        'pretrained-against-norm-bounding',
        'pretrained-against-neuroclip',
    }.issubset(origins):
        raise ValueError('white-box attack domain requires norm-bounding and neuroclip BRL origins')
    resolved = config or MNISTWhiteBoxMetaSGConfig()
    defender = _agent(
        obs_dim=environment_factory.defender_observation_dim,
        role='defender',
        seed=seed,
        config=resolved,
    )
    attackers = {}
    for index, (label, snapshot) in enumerate(sorted(attack_domain.snapshots.items())):
        attacker = _agent(
            obs_dim=environment_factory.attacker_observation_dim,
            role='attacker',
            seed=seed + index + 1,
            config=resolved,
        )
        attacker.restore(snapshot)
        attackers[label] = attacker
    sampler = UniformAttackTypeSampler(tuple(sorted(attackers)), seed=seed + 10_000)
    runner = _Algorithm1WhiteBoxRunner(
        environment_factory=environment_factory,
        config=resolved,
        support_seed=support_seed,
        query_seeds=query_seeds,
    )
    algorithm1 = runner.run(
        defender=defender,
        attackers=attackers,
        sample_tasks=sampler,
    )
    result = MNISTWhiteBoxMetaSGResult(
        algorithm1=algorithm1,
        defender=defender,
        attackers=attackers,
        support_seeds=tuple(runner.support_seeds),
        trajectory_count=len(runner.support_seeds),
        trajectories_per_update=runner.trajectories_per_update,
    )
    _write_manifest(
        output_dir=output_dir,
        result=result,
        config=resolved,
        origins=attack_domain.origins,
    )
    return result


def _agent(
    *,
    obs_dim: int,
    role: str,
    seed: int,
    config: MNISTWhiteBoxMetaSGConfig,
) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
        role=role,
        seed=seed,
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.policy_learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        policy_delay=config.policy_delay,
        target_policy_noise=config.target_policy_noise,
        noise_clip=config.noise_clip,
    )


def _write_manifest(
    *,
    output_dir: str | Path,
    result: MNISTWhiteBoxMetaSGResult,
    config: MNISTWhiteBoxMetaSGConfig,
    origins: Mapping[str, str],
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'manifest.json'
    payload = {
        'schema_version': 1,
        'protocol': result.protocol,
        'algorithm': 'meta-sg-algorithm1-reptile',
        'defender_action_dim': result.defender_action_dim,
        'attacker_action_dim': result.attacker_action_dim,
        'config': asdict(config),
        'attack_origins': dict(sorted(origins.items())),
        'support_seeds': list(result.support_seeds),
        'trajectory_count': result.trajectory_count,
        'defender_fingerprint': result.defender.fingerprint(),
        'attacker_fingerprints': {
            label: policy.fingerprint()
            for label, policy in sorted(result.attackers.items())
        },
        'query_data_used_for_training': False,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix='.manifest.', suffix='.tmp', dir=directory,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write('\n')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
