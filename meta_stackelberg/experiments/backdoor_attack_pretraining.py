"""BRL policy pre-training against fixed backdoor defenses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.attack_pretraining import (
    AttackPolicyPretrainer,
    AttackPolicyPretrainingConfig,
    AttackPolicyPretrainingResult,
    AttackTypeDomainPretrainingResult,
    build_attack_type_domain,
)
from meta_stackelberg.experiments.attack_pretraining_checkpoint import (
    load_attack_pretraining_checkpoint,
    save_attack_pretraining_checkpoint,
)
from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
)
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.clipping import ClippedAggregator
from meta_stackelberg.security.defenses.neuroclip import NeuroClipCopy


@dataclass(frozen=True)
class BackdoorAttackPretrainingTask:
    label: str
    defense: str
    clip_radius: float | None = None
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError('backdoor pre-training label must not be empty')
        fixed_backdoor_pretraining_defense(
            self.defense,
            clip_radius=self.clip_radius,
            epsilon=self.epsilon,
        )

    @property
    def origin(self) -> str:
        return f'pretrained-against-{self.defense.lower()}'


class FixedBackdoorPretrainingDefense:
    """One fixed aggregation/post-defense pair for BRL pre-training."""

    def __init__(
        self,
        *,
        defense: str,
        aggregator,
        clip_radius: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        self.defense = defense
        self.aggregator = aggregator
        self.clip_radius = clip_radius
        self.epsilon = epsilon
        values: dict[str, object] = {'defense': defense}
        if clip_radius is not None:
            values['clip_radius'] = clip_radius
        if epsilon is not None:
            values['epsilon'] = epsilon
        self.spec: Mapping[str, object] = MappingProxyType(values)

    def aggregate(self, updates: Sequence[ClientUpdate]):
        return self.aggregator.aggregate(updates)

    def post_defense_factory(self, model, action_epsilon):
        del action_epsilon
        if self.defense == 'neuroclip':
            return NeuroClipCopy(model, self.epsilon)
        return model

    def pretraining_spec(self) -> dict[str, object]:
        return dict(self.spec)


def fixed_backdoor_pretraining_defense(
    defense: str,
    *,
    clip_radius: float | None = None,
    epsilon: float | None = None,
) -> FixedBackdoorPretrainingDefense:
    normalized = defense.lower()
    if normalized == 'norm-bounding':
        radius = _positive(clip_radius, 'clip_radius')
        if epsilon is not None:
            raise ValueError('norm-bounding does not use epsilon')
        return FixedBackdoorPretrainingDefense(
            defense=normalized,
            aggregator=ClippedAggregator(FedAvg(), radius),
            clip_radius=radius,
        )
    if normalized == 'neuroclip':
        limit = _positive(epsilon, 'epsilon')
        if clip_radius is not None:
            raise ValueError('neuroclip does not use clip_radius')
        return FixedBackdoorPretrainingDefense(
            defense=normalized,
            aggregator=FedAvg(),
            epsilon=limit,
        )
    raise ValueError('backdoor defense must be norm-bounding or neuroclip')


def pretrain_backdoor_attack_type_domain(
    *,
    config: AttackPolicyPretrainingConfig,
    paper: PaperMetaSGConfig,
    environment_factory: PaperMNISTBackdoorEnvironmentFactory,
    tasks: tuple[BackdoorAttackPretrainingTask, ...],
    hidden_sizes: tuple[int, ...],
    seed: int,
    checkpoint_directory: str | Path | None = None,
    checkpoint_interval: int = 25,
    resume_checkpoints: bool = False,
) -> AttackTypeDomainPretrainingResult:
    if not isinstance(environment_factory, PaperMNISTBackdoorEnvironmentFactory):
        raise TypeError('environment_factory must be PaperMNISTBackdoorEnvironmentFactory')
    if not tasks or len({task.label for task in tasks}) != len(tasks):
        raise ValueError('backdoor tasks must have unique non-empty labels')
    if not hidden_sizes or any(value <= 0 for value in hidden_sizes):
        raise ValueError('hidden_sizes must contain positive integers')
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory else None
    if resume_checkpoints and checkpoint_root is None:
        raise ValueError('resume_checkpoints requires checkpoint_directory')
    if checkpoint_root is not None and checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be positive')
    defender = _agent(
        paper=paper,
        obs_dim=environment_factory.defender_observation_dim,
        role='defender',
        seed=seed + 1,
        hidden_sizes=hidden_sizes,
    )
    trainer = AttackPolicyPretrainer(config)
    results: list[AttackPolicyPretrainingResult] = []
    for index, task in enumerate(tasks):
        attacker = _agent(
            paper=paper,
            obs_dim=environment_factory.attacker_observation_dim,
            role='attacker',
            seed=seed + 10 + index,
            hidden_sizes=hidden_sizes,
        )
        env = environment_factory.make(
            seed=seed + 100 + index,
            horizon=config.fl_rounds,
            task_id=f'backdoor-attack-pretraining:{task.label}',
        )
        defense = fixed_backdoor_pretraining_defense(
            task.defense,
            clip_radius=task.clip_radius,
            epsilon=task.epsilon,
        )
        checkpoint_path = (
            checkpoint_root / f'{task.label}.pt'
            if checkpoint_root is not None
            else None
        )
        callback = (
            lambda checkpoint, path=checkpoint_path: save_attack_pretraining_checkpoint(
                path, checkpoint,
            )
            if checkpoint_path is not None
            else None
        )
        if resume_checkpoints and checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = load_attack_pretraining_checkpoint(checkpoint_path)
            if (checkpoint.label, checkpoint.origin) != (task.label, task.origin):
                raise ValueError('checkpoint backdoor task identity mismatch')
            result = trainer.resume(
                checkpoint=checkpoint,
                env=env,
                defender=defender,
                attacker=attacker,
                aggregator=defense,
                checkpoint_callback=callback,
                checkpoint_interval=checkpoint_interval,
            )
        else:
            result = trainer.train(
                label=task.label,
                origin=task.origin,
                env=env,
                defender=defender,
                attacker=attacker,
                aggregator=defense,
                replay_seed=seed + 1_000 + index,
                checkpoint_callback=callback,
                checkpoint_interval=checkpoint_interval,
            )
        results.append(result)
    completed = tuple(results)
    return AttackTypeDomainPretrainingResult(
        domain=build_attack_type_domain(completed),
        tasks=completed,
        total_fl_round_count=sum(item.fl_round_count for item in completed),
    )


def _agent(
    *,
    paper: PaperMetaSGConfig,
    obs_dim: int,
    role: str,
    seed: int,
    hidden_sizes: tuple[int, ...],
) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim,
        action_dim=3,
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


def _positive(value: float | None, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f'{name} must be provided')
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return result
