"""Trajectory collection primitives for paper-aligned per-round TD3 policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.agents.td3.config import (
    ScaledMetaSGConfig,
    ScaledOnlineAdaptationConfig,
)
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv, PaperRoundStep
from meta_stackelberg.experiments.scaled_training_checkpoint import (
    ScaledTrainingCheckpoint,
    load_scaled_training_checkpoint,
    save_scaled_training_checkpoint,
)
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
from meta_stackelberg.stackelberg.policy_online import (
    PolicyOnlineAdaptationResult,
    PolicyOnlineAdaptationRunner,
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
    fl_round_count: int | None = None

    def __post_init__(self) -> None:
        round_count = len(self.steps) if self.fl_round_count is None else self.fl_round_count
        if (
            isinstance(round_count, bool)
            or not isinstance(round_count, int)
            or round_count <= 0
        ):
            raise ValueError('trajectory FL round count must be positive')
        if self.steps and len(self.steps) != round_count:
            raise ValueError('retained trajectory steps must match FL round count')
        object.__setattr__(self, 'fl_round_count', round_count)

    @property
    def mean_defender_reward(self) -> float:
        return self.defender_return / self.fl_round_count

    @property
    def mean_attacker_reward(self) -> float:
        return self.attacker_return / self.fl_round_count


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


@dataclass(frozen=True)
class PaperOnlineTrainingResult:
    adaptation: PolicyOnlineAdaptationResult
    support_seeds: tuple[int, ...]
    trajectory_count: int
    fl_round_count: int
    trajectories_per_update: int
    scale_provenance: str = 'scaled-online-training-v1'


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
        retain_steps: bool = True,
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
        defender_return = 0.0
        attacker_return = 0.0
        fl_round_count = 0
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
            defender_return += float(step.defender_reward.scalar)
            attacker_return += float(step.attacker_reward.scalar)
            fl_round_count += 1
            if retain_steps:
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
            defender_return,
            attacker_return,
            fl_round_count,
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
        protocol_signature: Mapping[str, object] | None = None,
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
        self._initial_support_seed = support_seed
        self._replay_serial = 0
        self._support_seeds = []
        self._query_seeds = frozenset(query_seeds)
        self.protocol_signature = dict(protocol_signature or {})
        required = max(config.td3_batch_size, config.learning_starts)
        self.trajectories_per_update = max(1, int(np.ceil(required / config.H)))

    def run(
        self,
        *,
        initial_defender: TD3Agent,
        initial_attackers: Mapping[object, TD3Agent],
        sample_tasks,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_interval: int = 1,
    ) -> ScaledPolicyTrainingResult:
        if resume and checkpoint_path is None:
            raise ValueError('resume requires checkpoint_path')
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval <= 0
        ):
            raise ValueError('checkpoint_interval must be positive')
        target = Path(checkpoint_path) if checkpoint_path is not None else None
        algorithm1_defender = initial_defender.clone()
        algorithm2_defender = initial_defender.clone()
        algorithm1_attackers = {
            task: attacker.clone() for task, attacker in initial_attackers.items()
        }
        algorithm2_responses = {
            task: attacker.clone() for task, attacker in initial_attackers.items()
        }
        signature = self._checkpoint_signature(
            initial_defender, initial_attackers,
        )
        algorithm1_history = []
        algorithm2_history = []
        algorithm1_start = 0
        algorithm2_start = 0
        if resume:
            checkpoint = load_scaled_training_checkpoint(target)
            if dict(checkpoint.config_signature) != signature:
                raise ValueError('scaled training checkpoint configuration mismatch')
            if set(checkpoint.algorithm1_attackers) != set(algorithm1_attackers):
                raise ValueError('scaled training checkpoint attack domain mismatch')
            algorithm1_defender.restore(checkpoint.algorithm1_defender)
            algorithm2_defender.restore(checkpoint.algorithm2_defender)
            for task, snapshot in checkpoint.algorithm1_attackers.items():
                algorithm1_attackers[task].restore(snapshot)
            algorithm1_history.extend(checkpoint.algorithm1_iterations)
            algorithm2_history.extend(checkpoint.algorithm2_iterations)
            algorithm1_start = checkpoint.algorithm1_completed
            algorithm2_start = checkpoint.algorithm2_completed
            self._support_seeds = list(checkpoint.support_seeds)
            self._next_support_seed = checkpoint.next_support_seed
            self._replay_serial = checkpoint.replay_serial

        def save_checkpoint(phase: str) -> None:
            if target is None:
                return
            save_scaled_training_checkpoint(
                target,
                ScaledTrainingCheckpoint(
                    phase=phase,
                    config_signature=signature,
                    algorithm1_completed=len(algorithm1_history),
                    algorithm2_completed=len(algorithm2_history),
                    algorithm1_defender=algorithm1_defender.snapshot(),
                    algorithm2_defender=algorithm2_defender.snapshot(),
                    algorithm1_attackers={
                        task: policy.snapshot()
                        for task, policy in algorithm1_attackers.items()
                    },
                    support_seeds=tuple(self._support_seeds),
                    next_support_seed=self._next_support_seed,
                    replay_serial=self._replay_serial,
                    algorithm1_iterations=tuple(algorithm1_history),
                    algorithm2_iterations=tuple(algorithm2_history),
                ),
            )

        def record_algorithm1(trace, defender, attackers) -> None:
            del defender, attackers
            algorithm1_history.append(trace)
            if (
                len(algorithm1_history) % checkpoint_interval == 0
                or len(algorithm1_history) == self.config.N_D
            ):
                save_checkpoint(
                    'algorithm2'
                    if len(algorithm1_history) == self.config.N_D
                    else 'algorithm1',
                )

        algorithm1_runner = PolicyMetaSGAlgorithm1(
            N_D=self.config.N_D,
            K=self.config.K,
            N_A=self.config.N_A,
            batch_size=self.config.td3_batch_size,
            eta=self.config.paper_reference.adaptation_step,
            kappa_A=self.config.paper_reference.kappa_attacker,
            kappa_D=self.config.paper_reference.kappa_defender,
        )
        if algorithm1_start < self.config.N_D:
            algorithm1_runner.run(
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
                start_iteration=algorithm1_start,
                iteration_callback=record_algorithm1,
            )
        algorithm1 = PolicyAlgorithm1Result(tuple(algorithm1_history))

        def record_algorithm2(trace, defender, responses) -> None:
            del defender, responses
            algorithm2_history.append(trace)
            if (
                len(algorithm2_history) % checkpoint_interval == 0
                or len(algorithm2_history) == self.config.T
            ):
                save_checkpoint(
                    'complete'
                    if len(algorithm2_history) == self.config.T
                    else 'algorithm2',
                )

        algorithm2_runner = PolicyMetaSGAlgorithm2(
            T=self.config.T,
            K=self.config.K,
            l=self.config.l,
            batch_size=self.config.td3_batch_size,
            kappa=self.config.paper_reference.kappa,
            meta_update_step=self.config.paper_reference.meta_update_step,
        )
        if algorithm2_start < self.config.T:
            algorithm2_runner.run(
                defender=algorithm2_defender,
                response_policies=algorithm2_responses,
                sample_tasks=sample_tasks,
                replay_factory=lambda task, iteration: self._new_replay('defender'),
                collect_task=lambda task, step, defender, attacker, replay, iteration: self._collect_updates(
                    task, defender, attacker, replay, 'defender', iteration,
                ),
                start_iteration=algorithm2_start,
                iteration_callback=record_algorithm2,
            )
        algorithm2 = PolicyAlgorithm2Result(tuple(algorithm2_history))
        if target is not None:
            save_checkpoint('complete')
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

    def _checkpoint_signature(
        self,
        initial_defender: TD3Agent,
        initial_attackers: Mapping[object, TD3Agent],
    ) -> dict[str, object]:
        return {
            'T': self.config.T,
            'K': self.config.K,
            'H': self.config.H,
            'l': self.config.l,
            'N_A': self.config.N_A,
            'N_D': self.config.N_D,
            'td3_batch_size': self.config.td3_batch_size,
            'learning_starts': self.config.learning_starts,
            'hidden_sizes': tuple(self.config.hidden_sizes),
            'replay_capacity': self.config.replay_capacity,
            'workers': self.config.workers,
            'untargeted_attackers': self.config.untargeted_attackers,
            'sample_size': self.config.sample_size,
            'defender_obs_dim': self.defender_obs_dim,
            'attacker_obs_dim': self.attacker_obs_dim,
            'initial_support_seed': self._initial_support_seed,
            'initial_defender': initial_defender.fingerprint(),
            'initial_attackers': tuple(sorted(
                (repr(task), policy.fingerprint())
                for task, policy in initial_attackers.items()
            )),
            'protocol_signature': self.protocol_signature,
        }

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
                retain_steps=False,
            )


class PaperOnlineAdaptationTrainingRunner:
    """Execute real H-round rollouts for the paper online adaptation budget."""

    def __init__(
        self,
        *,
        config: ScaledOnlineAdaptationConfig,
        env_factory,
        defender_obs_dim: int,
        attacker_obs_dim: int,
        support_seeds: tuple[int, ...],
    ) -> None:
        if not isinstance(config, ScaledOnlineAdaptationConfig):
            raise TypeError('config must be ScaledOnlineAdaptationConfig')
        self.config = config
        self.env_factory = env_factory
        self.defender_obs_dim = defender_obs_dim
        self.attacker_obs_dim = attacker_obs_dim
        self.support_seeds = support_seeds
        required = max(config.td3_batch_size, config.learning_starts)
        self.trajectories_per_update = max(
            1, int(np.ceil(required / config.online_H)),
        )
        required_seeds = config.online_steps * self.trajectories_per_update
        if len(support_seeds) != required_seeds or len(set(support_seeds)) != required_seeds:
            raise ValueError('online support seeds must exactly match trajectory budget')
        self._seed_cursor = 0
        self._replay_serial = 0

    def run(self, *, task, meta_defender, attacker) -> PaperOnlineTrainingResult:
        replay = self._new_replay('defender')
        result = PolicyOnlineAdaptationRunner(
            online_T=self.config.online_T,
            online_l=self.config.online_l,
            online_steps=self.config.online_steps,
            batch_size=self.config.td3_batch_size,
            adaptation_step=self.config.paper_reference.adaptation_step,
        ).run(
            meta_defender=meta_defender,
            attacker=attacker,
            replay=replay,
            collect_fresh=lambda defender, frozen, target, iteration, local, global_step: self._collect(
                task, defender, frozen, target, iteration,
            ),
        )
        if self._seed_cursor != len(self.support_seeds):
            raise RuntimeError('online trajectory seed budget was not exhausted exactly')
        return PaperOnlineTrainingResult(
            result,
            self.support_seeds,
            self._seed_cursor,
            self._seed_cursor * self.config.online_H,
            self.trajectories_per_update,
        )

    def _collect(self, task, defender, attacker, replay, generation):
        for _ in range(self.trajectories_per_update):
            seed = self.support_seeds[self._seed_cursor]
            self._seed_cursor += 1
            PaperTD3TrajectoryCollector().collect(
                env=self.env_factory(task, seed, self.config.online_H),
                defender=defender,
                attacker=attacker,
                defender_replay=replay,
                attacker_replay=self._new_replay('attacker'),
                generation=generation,
                deterministic=False,
                explore_role='defender',
                retain_steps=False,
            )

    def _new_replay(self, role):
        self._replay_serial += 1
        return TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=(self.defender_obs_dim if role == 'defender' else self.attacker_obs_dim),
            action_dim=3,
            role=role,
            seed=30_000_000 + self._replay_serial,
        )
