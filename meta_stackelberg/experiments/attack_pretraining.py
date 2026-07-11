"""Sequential TD3 pre-training for paper attack-type policy domains."""

from __future__ import annotations

from dataclasses import dataclass

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot, TD3UpdateStats
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, flatten_observation
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv
from meta_stackelberg.experiments.attack_domain import AttackTypeDomainSource
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


@dataclass(frozen=True)
class AttackPolicyPretrainingResult:
    label: str
    origin: str
    policy: TD3Snapshot
    fl_round_count: int
    replay_transition_count: int
    td3_update_count: int
    update_stats: tuple[TD3UpdateStats, ...]
    protocol: str = 'sequential-fl-round-td3-pretraining-v1'


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
    ) -> AttackPolicyPretrainingResult:
        if not label or not origin:
            raise ValueError('label and origin must not be empty')
        if defender.role != 'defender' or attacker.role != 'attacker':
            raise ValueError('policy roles do not match pre-training protocol')
        if env.state.round_index != 0 or env.horizon != self.config.fl_rounds:
            raise ValueError('environment must start at zero with fl_rounds horizon')
        if not callable(getattr(aggregator, 'aggregate', None)):
            raise TypeError('aggregator must provide aggregate(updates)')
        env.aggregator_factory = lambda action: aggregator
        replay = TD3ReplayBuffer(
            self.config.replay_capacity,
            obs_dim=attacker.obs_dim,
            action_dim=attacker.action_dim,
            role='attacker',
            seed=replay_seed,
        )
        defender_guard = defender.freeze_guard()
        pending_attacker = None
        stats: list[TD3UpdateStats] = []
        transition_count = 0

        while env.state.round_index < env.horizon:
            defender_observation = flatten_observation(
                env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
            )
            defender_action = defender.act(defender_observation, deterministic=True)
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
        )

    def _update_if_due(
        self,
        replay: TD3ReplayBuffer,
        transition_count: int,
        attacker: TD3Agent,
        stats: list[TD3UpdateStats],
    ) -> None:
        ready = max(self.config.learning_starts, self.config.batch_size)
        if len(replay) < ready or transition_count % self.config.train_freq != 0:
            return
        for _ in range(self.config.gradient_steps):
            stats.append(attacker.update(replay.sample(self.config.batch_size)))
