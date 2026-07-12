"""Sequential TD3 pre-training for paper attack-type policy domains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot, TD3UpdateStats
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
from meta_stackelberg.experiments.attack_pretraining_checkpoint import (
    AttackPretrainingCheckpoint,
    load_attack_pretraining_checkpoint,
    save_attack_pretraining_checkpoint,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)
from meta_stackelberg.federated.aggregation.coordinate_median import CoordinateMedian
from meta_stackelberg.security.defenses.clipping import ClippedAggregator
from meta_stackelberg.security.defenses.krum import Krum


@dataclass(frozen=True)
class AttackPolicyPretrainingConfig:
    fl_rounds: int = 300
    batch_size: int = 256
    learning_starts: int = 100
    train_freq: int = 1
    gradient_steps: int = 1
    replay_capacity: int = 1_000_000
    fixed_defender_raw_action: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @classmethod
    def from_paper(
        cls,
        paper: PaperMetaSGConfig,
    ) -> 'AttackPolicyPretrainingConfig':
        if not isinstance(paper, PaperMetaSGConfig):
            raise TypeError('paper must be PaperMetaSGConfig')
        return cls(
            fl_rounds=paper.rl_training_rounds,
            batch_size=paper.td3_batch_size,
            learning_starts=paper.learning_starts,
            train_freq=paper.train_freq,
            gradient_steps=paper.gradient_steps,
            replay_capacity=paper.replay_capacity,
        )

    def __post_init__(self) -> None:
        for name in (
            'fl_rounds', 'batch_size', 'learning_starts', 'train_freq',
            'gradient_steps', 'replay_capacity',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.batch_size > self.replay_capacity:
            raise ValueError('batch_size must not exceed replay_capacity')
        if self.learning_starts > self.replay_capacity:
            raise ValueError('learning_starts must not exceed replay_capacity')
        action = np.asarray(self.fixed_defender_raw_action, dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(
            (action < -1.0) | (action > 1.0)
        ):
            raise ValueError(
                'fixed_defender_raw_action must be finite within [-1,1]^3',
            )


@dataclass(frozen=True)
class AttackPolicyPretrainingResult:
    label: str
    origin: str
    policy: TD3Snapshot
    fl_round_count: int
    replay_transition_count: int
    td3_update_count: int
    update_stats: tuple[TD3UpdateStats, ...]
    fixed_defender_raw_action: tuple[float, float, float]
    protocol: str = 'sequential-fl-round-td3-pretraining-v1'


@dataclass(frozen=True)
class AttackPretrainingTaskSpec:
    label: str
    defense: str
    byzantine_count: int | None = None
    clip_radius: float | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError('attack pre-training task label must not be empty')
        fixed_pretraining_aggregator(
            self.defense,
            byzantine_count=self.byzantine_count,
            clip_radius=self.clip_radius,
        )

    @property
    def origin(self) -> str:
        return f'pretrained-against-{self.defense.lower()}'


@dataclass(frozen=True)
class AttackTypeDomainPretrainingResult:
    domain: AttackTypeDomainSource
    tasks: tuple[AttackPolicyPretrainingResult, ...]
    total_fl_round_count: int
    protocol: str = 'paper-attack-type-domain-pretraining-v1'


def build_attack_type_domain(
    results: tuple[AttackPolicyPretrainingResult, ...],
) -> AttackTypeDomainSource:
    if not results:
        raise ValueError('pre-training results must not be empty')
    labels = tuple(result.label for result in results)
    if len(set(labels)) != len(labels):
        raise ValueError('pre-training result labels must be unique')
    protocols = {result.protocol for result in results}
    if len(protocols) != 1:
        raise ValueError('pre-training result protocols must match')
    return AttackTypeDomainSource(
        {result.label: result.policy for result in results},
        {result.label: result.origin for result in results},
        protocol=protocols.pop(),
    )


def fixed_pretraining_aggregator(
    defense: str,
    *,
    byzantine_count: int | None = None,
    clip_radius: float | None = None,
):
    """Build the fixed defense used to create one attacker task type."""
    normalized = defense.lower()
    if normalized == 'krum':
        if byzantine_count is None:
            raise ValueError('Krum pre-training requires byzantine_count')
        if clip_radius is not None:
            raise ValueError('Krum pre-training does not use clip_radius')
        return Krum(byzantine_count)
    if normalized == 'clipmed':
        if clip_radius is None:
            raise ValueError('ClipMed pre-training requires clip_radius')
        if byzantine_count is not None:
            raise ValueError('ClipMed pre-training does not use byzantine_count')
        return ClippedAggregator(CoordinateMedian(), clip_radius)
    raise ValueError('defense must be krum or clipmed')


def _aggregator_spec(aggregator) -> dict[str, object]:
    custom = getattr(aggregator, 'pretraining_spec', None)
    if callable(custom):
        result = custom()
        if not isinstance(result, dict) or not result or 'defense' not in result:
            raise ValueError('custom pre-training defense spec is invalid')
        return result
    if isinstance(aggregator, Krum):
        return {
            'defense': 'krum',
            'byzantine_count': aggregator.byzantine_count,
        }
    if isinstance(aggregator, ClippedAggregator) and isinstance(
        aggregator.base, CoordinateMedian,
    ):
        return {
            'defense': 'clipmed',
            'clip_radius': aggregator.clip_radius,
        }
    raise ValueError('unsupported fixed pre-training aggregator')


def pretrain_attack_type_domain(
    *,
    config: AttackPolicyPretrainingConfig,
    paper: PaperMetaSGConfig,
    env_factory,
    tasks: tuple[AttackPretrainingTaskSpec, ...],
    hidden_sizes: tuple[int, ...],
    seed: int,
    checkpoint_directory: str | Path | None = None,
    checkpoint_interval: int = 25,
    resume_checkpoints: bool = False,
) -> AttackTypeDomainPretrainingResult:
    if not tasks or len({task.label for task in tasks}) != len(tasks):
        raise ValueError('pre-training tasks must have unique non-empty labels')
    if not hidden_sizes or any(value <= 0 for value in hidden_sizes):
        raise ValueError('hidden_sizes must contain positive integers')
    probe = env_factory(seed, config.fl_rounds, 'attack-pretraining-probe')
    defender_observation = probe.defender_observation()
    defender_obs_dim = len(flatten_observation(
        defender_observation, DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_observation = probe.observation_encoder.attacker_observation(
        defender_observation,
        malicious_count=0,
        defender_raw_action=np.zeros(3, dtype=np.float32),
    )
    attacker_obs_dim = len(flatten_observation(
        attacker_observation, ATTACKER_OBSERVATION_KEYS,
    ))
    defender = _pretraining_agent(
        paper, defender_obs_dim, 'defender', seed + 1, hidden_sizes,
    )
    trainer = AttackPolicyPretrainer(config)
    checkpoint_root = (
        Path(checkpoint_directory) if checkpoint_directory is not None else None
    )
    if resume_checkpoints and checkpoint_root is None:
        raise ValueError('resume_checkpoints requires checkpoint_directory')
    if checkpoint_root is not None and (
        isinstance(checkpoint_interval, bool)
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
    ):
        raise ValueError('checkpoint_interval must be a positive integer')
    results = []
    for index, task in enumerate(tasks):
        attacker = _pretraining_agent(
            paper, attacker_obs_dim, 'attacker', seed + 10 + index,
            hidden_sizes,
        )
        env = env_factory(
            seed + 100 + index,
            config.fl_rounds,
            f'attack-pretraining-{task.label}',
        )
        aggregator = fixed_pretraining_aggregator(
            task.defense,
            byzantine_count=task.byzantine_count,
            clip_radius=task.clip_radius,
        )
        checkpoint_path = (
            checkpoint_root / f'{task.label}.pt'
            if checkpoint_root is not None else None
        )
        checkpoint_callback = (
            lambda checkpoint, path=checkpoint_path: save_attack_pretraining_checkpoint(
                path, checkpoint,
            )
            if checkpoint_path is not None else None
        )
        if resume_checkpoints and checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = load_attack_pretraining_checkpoint(checkpoint_path)
            if (checkpoint.label, checkpoint.origin) != (task.label, task.origin):
                raise ValueError('checkpoint attack task identity mismatch')
            result = trainer.resume(
                checkpoint=checkpoint,
                env=env,
                defender=defender,
                attacker=attacker,
                aggregator=aggregator,
                checkpoint_callback=checkpoint_callback,
                checkpoint_interval=checkpoint_interval,
            )
        else:
            result = trainer.train(
                label=task.label,
                origin=task.origin,
                env=env,
                defender=defender,
                attacker=attacker,
                aggregator=aggregator,
                replay_seed=seed + 1_000 + index,
                checkpoint_callback=checkpoint_callback,
                checkpoint_interval=checkpoint_interval,
            )
        results.append(result)
    values = tuple(results)
    return AttackTypeDomainPretrainingResult(
        build_attack_type_domain(values),
        values,
        sum(result.fl_round_count for result in values),
    )


def _pretraining_agent(
    paper: PaperMetaSGConfig,
    obs_dim: int,
    role: str,
    seed: int,
    hidden_sizes: tuple[int, ...],
) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=paper.attacker_action_dim,
        role=role,
        seed=seed,
        hidden_sizes=hidden_sizes,
        learning_rate=paper.policy_learning_rate,
        gamma=paper.gamma,
        tau=paper.tau,
        policy_delay=paper.policy_delay,
        target_policy_noise=paper.target_policy_noise,
        noise_clip=paper.noise_clip,
    )


class AttackPolicyPretrainer:
    """Train one attacker over a single, sequential FL trajectory."""

    def __init__(self, config: AttackPolicyPretrainingConfig) -> None:
        if not isinstance(config, AttackPolicyPretrainingConfig):
            raise TypeError('config must be AttackPolicyPretrainingConfig')
        self.config = config

    def train(
        self,
        *,
        label: str,
        origin: str,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        aggregator,
        replay_seed: int,
        checkpoint_callback=None,
        checkpoint_interval: int = 1,
    ) -> AttackPolicyPretrainingResult:
        replay = self._new_replay(attacker, replay_seed)
        result = self._execute(
            label=label,
            origin=origin,
            env=env,
            defender=defender,
            attacker=attacker,
            aggregator=aggregator,
            replay=replay,
            pending_attacker=None,
            transition_count=0,
            stats=[],
            stop_after_round=None,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
        )
        if not isinstance(result, AttackPolicyPretrainingResult):
            raise RuntimeError('uninterrupted pre-training did not complete')
        return result

    def pause(
        self,
        *,
        label: str,
        origin: str,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        aggregator,
        replay_seed: int,
        stop_after_round: int,
    ) -> AttackPretrainingCheckpoint:
        if (
            isinstance(stop_after_round, bool)
            or not isinstance(stop_after_round, int)
            or stop_after_round <= 0
            or stop_after_round >= self.config.fl_rounds
        ):
            raise ValueError('stop_after_round must be within [1, fl_rounds)')
        result = self._execute(
            label=label,
            origin=origin,
            env=env,
            defender=defender,
            attacker=attacker,
            aggregator=aggregator,
            replay=self._new_replay(attacker, replay_seed),
            pending_attacker=None,
            transition_count=0,
            stats=[],
            stop_after_round=stop_after_round,
            checkpoint_callback=None,
            checkpoint_interval=1,
        )
        if not isinstance(result, AttackPretrainingCheckpoint):
            raise RuntimeError('pre-training completed before pause boundary')
        return result

    def resume(
        self,
        *,
        checkpoint: AttackPretrainingCheckpoint,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        aggregator,
        checkpoint_callback=None,
        checkpoint_interval: int = 1,
    ) -> AttackPolicyPretrainingResult:
        if not isinstance(checkpoint, AttackPretrainingCheckpoint):
            raise TypeError('checkpoint must be AttackPretrainingCheckpoint')
        if dict(checkpoint.config) != asdict(self.config):
            raise ValueError('checkpoint pre-training config mismatch')
        if checkpoint.defender_fingerprint != defender.fingerprint():
            raise ValueError('checkpoint frozen Defender fingerprint mismatch')
        if checkpoint.aggregator_spec != _aggregator_spec(aggregator):
            raise ValueError('checkpoint fixed aggregator mismatch')
        if env.horizon != self.config.fl_rounds or env.state.round_index != 0:
            raise ValueError('resume environment must be fresh with fl_rounds horizon')
        attacker.restore(checkpoint.attacker_snapshot)
        replay = TD3ReplayBuffer(
            checkpoint.replay_snapshot.capacity,
            obs_dim=checkpoint.replay_snapshot.obs_dim,
            action_dim=checkpoint.replay_snapshot.action_dim,
            role='attacker',
            seed=0,
        )
        replay.restore(checkpoint.replay_snapshot)
        env.state = checkpoint.round_state
        env.rng.restore(checkpoint.round_state.random_snapshot)
        env.observed_max_norm = checkpoint.observed_max_norm
        env._last_epsilon = checkpoint.last_epsilon
        result = self._execute(
            label=checkpoint.label,
            origin=checkpoint.origin,
            env=env,
            defender=defender,
            attacker=attacker,
            aggregator=aggregator,
            replay=replay,
            pending_attacker=(
                checkpoint.pending_observation.copy(),
                checkpoint.pending_action.copy(),
                checkpoint.pending_reward,
            ),
            transition_count=checkpoint.transition_count,
            stats=list(checkpoint.update_stats),
            stop_after_round=None,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
        )
        if not isinstance(result, AttackPolicyPretrainingResult):
            raise RuntimeError('resumed pre-training did not complete')
        return result

    def _new_replay(
        self,
        attacker: TD3Agent,
        replay_seed: int,
    ) -> TD3ReplayBuffer:
        return TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=attacker.obs_dim,
            action_dim=attacker.action_dim,
            role='attacker',
            seed=replay_seed,
        )

    def _execute(
        self,
        *,
        label: str,
        origin: str,
        env: PaperBSMGEnv,
        defender: TD3Agent,
        attacker: TD3Agent,
        aggregator,
        replay: TD3ReplayBuffer,
        pending_attacker,
        transition_count: int,
        stats: list[TD3UpdateStats],
        stop_after_round: int | None,
        checkpoint_callback,
        checkpoint_interval: int,
    ) -> AttackPolicyPretrainingResult | AttackPretrainingCheckpoint:
        if not label or not origin:
            raise ValueError('label and origin must not be empty')
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('policy roles do not match pre-training protocol')
        if env.horizon != self.config.fl_rounds:
            raise ValueError('environment horizon must equal fl_rounds')
        if not callable(getattr(aggregator, 'aggregate', None)):
            raise TypeError('aggregator must provide aggregate(updates)')
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval <= 0
        ):
            raise ValueError('checkpoint_interval must be a positive integer')
        env.aggregator_factory = lambda action: aggregator
        post_defense = getattr(aggregator, 'post_defense_factory', None)
        env.post_defense_factory = (
            post_defense
            if callable(post_defense)
            else lambda model, epsilon: model
        )
        defender_guard = defender.freeze_guard()

        while env.state.round_index < env.horizon:
            defender_action = np.asarray(
                self.config.fixed_defender_raw_action, dtype=np.float32,
            )
            pending = env.begin_round(defender_action)
            attacker_observation = flatten_observation(
                pending.attacker_observation, ATTACKER_OBSERVATION_KEYS,
            )
            if pending_attacker is not None:
                old_observation, old_action, old_reward = pending_attacker
                replay.add(
                    old_observation, old_action, old_reward,
                    attacker_observation, False,
                    generation=0, role='attacker',
                )
                transition_count += 1
                self._update_if_due(replay, transition_count, attacker, stats)
            attacker_action = attacker.act(attacker_observation, deterministic=False)
            step = env.finish_round(attacker_action)
            pending_attacker = (
                attacker_observation, attacker_action, step.attacker_reward.scalar,
            )
            defender_guard.verify()
            should_stop = stop_after_round == env.state.round_index
            should_checkpoint = (
                checkpoint_callback is not None
                and env.state.round_index % checkpoint_interval == 0
                and env.state.round_index < env.horizon
            )
            if should_stop or should_checkpoint:
                checkpoint = self._checkpoint(
                    label, origin, env, attacker, replay, pending_attacker,
                    transition_count, stats, defender_guard.fingerprint,
                    aggregator,
                )
                if should_checkpoint:
                    checkpoint_callback(checkpoint)
                if should_stop:
                    return checkpoint

        if pending_attacker is not None:
            observation, action, reward = pending_attacker
            replay.add(
                observation, action, reward, observation, True,
                generation=0, role='attacker',
            )
            transition_count += 1
            self._update_if_due(replay, transition_count, attacker, stats)
        defender_guard.verify()
        return AttackPolicyPretrainingResult(
            label,
            origin,
            attacker.snapshot(),
            env.state.round_index,
            transition_count,
            len(stats),
            tuple(stats),
            self.config.fixed_defender_raw_action,
        )

    def _checkpoint(
        self,
        label,
        origin,
        env,
        attacker,
        replay,
        pending_attacker,
        transition_count,
        stats,
        defender_fingerprint,
        aggregator,
    ) -> AttackPretrainingCheckpoint:
        return AttackPretrainingCheckpoint(
            label=label,
            origin=origin,
            config=asdict(self.config),
            aggregator_spec=_aggregator_spec(aggregator),
            round_state=env.state,
            observed_max_norm=env.observed_max_norm,
            last_epsilon=env._last_epsilon,
            pending_observation=pending_attacker[0],
            pending_action=pending_attacker[1],
            pending_reward=pending_attacker[2],
            attacker_snapshot=attacker.snapshot(),
            replay_snapshot=replay.snapshot(),
            transition_count=transition_count,
            update_stats=tuple(stats),
            defender_fingerprint=defender_fingerprint,
        )

    def _update_if_due(
        self,
        replay: TD3ReplayBuffer,
        transition_count: int,
        attacker: TD3Agent,
        stats: list[TD3UpdateStats],
    ) -> None:
        if (
            len(replay) < self.config.learning_starts
            or transition_count % self.config.train_freq != 0
        ):
            return
        for _ in range(self.config.gradient_steps):
            stats.append(attacker.update(replay.sample(
                self.config.batch_size,
                replace=self.config.batch_size > len(replay),
            )))
