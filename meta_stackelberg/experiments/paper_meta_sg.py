"""Trajectory collection primitives for paper-aligned per-round TD3 policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.agents.td3.config import ScaledMetaSGConfig
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv, PaperRoundStep
from meta_stackelberg.security.attacks.rl_action import RLAttackActionCodec
from meta_stackelberg.security.defenses.paper_action import PaperDefenderActionCodec
from meta_stackelberg.stackelberg.algorithm1 import Algorithm1Result
from meta_stackelberg.stackelberg.algorithm2 import Algorithm2Result
from meta_stackelberg.stackelberg.policy_algorithm1 import (
    PolicyAlgorithm1Result,
    PolicyMetaSGAlgorithm1,
)
from meta_stackelberg.stackelberg.policy_algorithm2 import (
    PolicyAlgorithm2Result,
    PolicyMetaSGAlgorithm2,
)


DEFENDER_OBSERVATION_KEYS = ('model_tail', 'round_progress')
ATTACKER_OBSERVATION_KEYS = (
    'model_tail', 'round_progress', 'malicious_count', 'defender_action',
)


@dataclass(frozen=True)
class ActionParameterRecord:
    role: str
    parameter: str
    low: float | int | str
    high: float | int | str
    source: str
    deviation: str


def action_parameter_ledger() -> tuple[ActionParameterRecord, ...]:
    defender = PaperDefenderActionCodec()
    attacker = RLAttackActionCodec()
    return (
        ActionParameterRecord(
            'defender', 'alpha', defender.alpha_min, 'observed_max_norm',
            'paper-semantic-with-numerical-floor',
            'paper uses (0,max sampled norm]; implementation adds 1e-6 floor',
        ),
        ActionParameterRecord(
            'defender', 'beta', 0.0, defender.beta_max,
            'implementation-declared',
            'paper states beta in [0,1); valid symmetric trimming requires beta<0.5 and codec caps 0.45',
        ),
        ActionParameterRecord(
            'defender', 'epsilon', defender.epsilon_min, defender.epsilon_max,
            'implementation-declared',
            'paper names NeuroClip epsilon but does not publish decoder bounds',
        ),
        ActionParameterRecord(
            'attacker', 'gamma', attacker.gamma_min, attacker.gamma_max,
            'rl-attacker-compatible-declared',
            'Meta-SG paper does not publish the continuous gamma decoder bounds',
        ),
        ActionParameterRecord(
            'attacker', 'local_steps', attacker.local_steps_min,
            attacker.local_steps_max, 'rl-attacker-compatible-declared',
            'Meta-SG paper does not publish the integer E decoder bounds',
        ),
        ActionParameterRecord(
            'attacker', 'stealth_lambda', attacker.stealth_min,
            attacker.stealth_max, 'rl-attacker-compatible-declared',
            'Meta-SG paper does not publish the lambda decoder bounds',
        ),
    )


@dataclass(frozen=True)
class PaperTrajectory:
    generation: int
    steps: tuple[PaperRoundStep, ...]
    defender_return: float
    attacker_return: float


@dataclass(frozen=True)
class ScaledConformanceResult:
    passed: bool
    checks: tuple[tuple[str, bool, int, int], ...]
    scale_provenance: str


@dataclass(frozen=True)
class ScaledPolicyTrainingResult:
    algorithm1: PolicyAlgorithm1Result
    algorithm2: PolicyAlgorithm2Result
    algorithm1_defender: TD3Agent
    algorithm2_defender: TD3Agent
    algorithm1_attackers: Mapping[object, TD3Agent]
    support_seeds: tuple[int, ...]
    trajectory_count: int
    trajectories_per_update: int
    scale_provenance: str = 'scaled-training-conformance-v1'


def evaluate_scaled_conformance(
    *,
    config: ScaledMetaSGConfig,
    algorithm1: Algorithm1Result,
    algorithm2: Algorithm2Result,
    trajectory: PaperTrajectory,
) -> ScaledConformanceResult:
    """Check paper parameter meanings against immutable execution traces."""
    counts = {
        'algorithm1_leader_iterations': len(algorithm1.iterations),
        'algorithm1_attacker_updates': sum(
            event.kind == 'attacker_update' for event in algorithm1.events
        ),
        'algorithm1_leader_updates': sum(
            event.kind == 'leader_update' for event in algorithm1.events
        ),
        'algorithm2_meta_iterations': len(algorithm2.iterations),
        'algorithm2_task_adaptations': sum(
            event.kind == 'adapt_task' for event in algorithm2.events
        ),
        'algorithm2_meta_updates': sum(
            event.kind == 'meta_update' for event in algorithm2.events
        ),
        'trajectory_fl_rounds': len(trajectory.steps),
        'defender_3d_actions': sum(
            step.defender_raw_action.shape == (3,) for step in trajectory.steps
        ),
        'attacker_3d_actions': sum(
            step.attacker_raw_action.shape == (3,) for step in trajectory.steps
        ),
    }
    expected = {
        'algorithm1_leader_iterations': config.N_D,
        'algorithm1_attacker_updates': config.N_D * config.K * config.N_A,
        'algorithm1_leader_updates': config.N_D,
        'algorithm2_meta_iterations': config.T,
        'algorithm2_task_adaptations': config.T * config.K * config.l,
        'algorithm2_meta_updates': config.T,
        'trajectory_fl_rounds': config.H,
        'defender_3d_actions': config.H,
        'attacker_3d_actions': config.H,
    }
    checks = tuple(
        (name, counts[name] == wanted, counts[name], wanted)
        for name, wanted in expected.items()
    )
    return ScaledConformanceResult(
        all(check[1] for check in checks), checks, config.scale_provenance,
    )


class PaperTD3TrajectoryCollector:
    """Collect one complete H-round trajectory with both policies acting each round."""

    def collect(
        self,
        *,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        defender_replay: TD3ReplayBuffer,
        attacker_replay: TD3ReplayBuffer,
        generation: int,
        deterministic: bool = False,
        explore_role: str | None = None,
    ) -> PaperTrajectory:
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('trajectory policy roles do not match protocol')
        if defender_replay.role != 'defender' or attacker_replay.role != 'attacker':
            raise ValueError('trajectory replay roles do not match protocol')
        if explore_role not in {None, 'defender', 'attacker'}:
            raise ValueError('explore_role must be defender, attacker or None')
        defender_deterministic = deterministic or explore_role == 'attacker'
        attacker_deterministic = deterministic or explore_role == 'defender'
        steps = []
        pending_attacker = None
        while env.state.round_index < env.horizon:
            defender_obs = flatten_observation(
                env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
            )
            defender_action = defender.act(
                defender_obs, deterministic=defender_deterministic,
            )
            pending = env.begin_round(defender_action)
            attacker_obs = flatten_observation(
                pending.attacker_observation, ATTACKER_OBSERVATION_KEYS,
            )
            if pending_attacker is not None:
                old_obs, old_action, old_reward = pending_attacker
                attacker_replay.add(
                    old_obs, old_action, old_reward, attacker_obs, False,
                    generation=generation, role='attacker',
                )
            attacker_action = attacker.act(
                attacker_obs, deterministic=attacker_deterministic,
            )
            step = env.finish_round(attacker_action)
            next_defender_obs = flatten_observation(
                env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
            )
            defender_replay.add(
                defender_obs, defender_action, step.defender_reward.scalar,
                next_defender_obs, step.done,
                generation=generation, role='defender',
            )
            pending_attacker = (
                attacker_obs, attacker_action, step.attacker_reward.scalar,
            )
            steps.append(step)
        if pending_attacker is not None:
            obs, action, reward = pending_attacker
            attacker_replay.add(
                obs, action, reward, obs, True,
                generation=generation, role='attacker',
            )
        return PaperTrajectory(
            generation,
            tuple(steps),
            float(sum(step.defender_reward.scalar for step in steps)),
            float(sum(step.attacker_reward.scalar for step in steps)),
        )


class ScaledPaperMetaSGTrainingRunner:
    """Wire real H-round rollouts into both paper policy algorithms."""

    def __init__(
        self,
        *,
        config: ScaledMetaSGConfig,
        env_factory,
        defender_obs_dim: int,
        attacker_obs_dim: int,
        support_seed: int,
        query_seeds: tuple[int, ...] = (),
    ) -> None:
        if not isinstance(config, ScaledMetaSGConfig):
            raise TypeError('config must be ScaledMetaSGConfig')
        if support_seed < 0:
            raise ValueError('support_seed must be non-negative')
        self.config = config
        self.env_factory = env_factory
        self.defender_obs_dim = defender_obs_dim
        self.attacker_obs_dim = attacker_obs_dim
        self._next_support_seed = support_seed
        self._replay_serial = 0
        self._support_seeds = []
        self._query_seeds = frozenset(query_seeds)
        required = max(config.td3_batch_size, config.learning_starts)
        self.trajectories_per_update = max(1, int(np.ceil(required / config.H)))

    def run(
        self,
        *,
        initial_defender: TD3Agent,
        initial_attackers: Mapping[object, TD3Agent],
        sample_tasks,
    ) -> ScaledPolicyTrainingResult:
        algorithm1_defender = initial_defender.clone()
        algorithm2_defender = initial_defender.clone()
        algorithm1_attackers = {
            task: attacker.clone() for task, attacker in initial_attackers.items()
        }
        algorithm2_responses = {
            task: attacker.clone() for task, attacker in initial_attackers.items()
        }
        algorithm1 = PolicyMetaSGAlgorithm1(
            N_D=self.config.N_D,
            K=self.config.K,
            N_A=self.config.N_A,
            batch_size=self.config.td3_batch_size,
            eta=self.config.paper_reference.adaptation_step,
            kappa_A=self.config.paper_reference.kappa_attacker,
            kappa_D=self.config.paper_reference.kappa_defender,
        ).run(
            defender=algorithm1_defender,
            attackers=algorithm1_attackers,
            sample_tasks=sample_tasks,
            replay_factory=self._algorithm1_replay,
            collect_adaptation=lambda task, defender, attacker, replay, iteration: self._collect_updates(
                task, defender, attacker, replay, 'defender', iteration,
            ),
            collect_response=lambda task, step, defender, attacker, replay, iteration: self._collect_updates(
                task, defender, attacker, replay, 'attacker', iteration,
            ),
            collect_leader=lambda task, defender, attacker, replay, iteration: self._collect_updates(
                task, defender, attacker, replay, 'defender', iteration,
            ),
            independent_attacker_objective=lambda task, policy: float(
                np.mean(policy.act(
                    np.zeros(self.attacker_obs_dim, dtype=np.float32),
                    deterministic=True,
                ))
            ),
        )
        algorithm2 = PolicyMetaSGAlgorithm2(
            T=self.config.T,
            K=self.config.K,
            l=self.config.l,
            batch_size=self.config.td3_batch_size,
            kappa=self.config.paper_reference.kappa,
            meta_update_step=self.config.paper_reference.meta_update_step,
        ).run(
            defender=algorithm2_defender,
            response_policies=algorithm2_responses,
            sample_tasks=sample_tasks,
            replay_factory=lambda task, iteration: self._new_replay('defender'),
            collect_task=lambda task, step, defender, attacker, replay, iteration: self._collect_updates(
                task, defender, attacker, replay, 'defender', iteration,
            ),
        )
        return ScaledPolicyTrainingResult(
            algorithm1,
            algorithm2,
            algorithm1_defender,
            algorithm2_defender,
            algorithm1_attackers,
            tuple(self._support_seeds),
            len(self._support_seeds),
            self.trajectories_per_update,
        )

    def _algorithm1_replay(self, task, role, phase, iteration):
        del task, phase, iteration
        return self._new_replay(role)

    def _new_replay(self, role: str) -> TD3ReplayBuffer:
        self._replay_serial += 1
        obs_dim = self.defender_obs_dim if role == 'defender' else self.attacker_obs_dim
        return TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=obs_dim,
            action_dim=3,
            role=role,
            seed=10_000_000 + self._replay_serial,
        )

    def _collect_updates(
        self,
        task,
        defender: TD3Agent,
        attacker: TD3Agent,
        target_replay: TD3ReplayBuffer,
        target_role: str,
        iteration: int,
    ) -> None:
        for _ in range(self.trajectories_per_update):
            seed = self._next_support_seed
            self._next_support_seed += 1
            if seed in self._query_seeds:
                raise RuntimeError('support trajectory seed overlaps held-out query seed')
            self._support_seeds.append(seed)
            env = self.env_factory(task, seed, self.config.H)
            defender_replay = (
                target_replay if target_role == 'defender' else self._new_replay('defender')
            )
            attacker_replay = (
                target_replay if target_role == 'attacker' else self._new_replay('attacker')
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
