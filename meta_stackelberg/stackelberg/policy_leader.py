"""Concrete TD3 leader update against frozen policy-level best responses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3UpdateStats
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer


@dataclass(frozen=True)
class PolicyLeaderTask:
    task_id: object
    adapted_defender: TD3Agent
    best_response: TD3Agent
    defender_replay: TD3ReplayBuffer


@dataclass(frozen=True)
class PolicyLeaderTaskUpdate:
    task_id: object
    best_response_fingerprint: str
    adapted_defender_fingerprint: str
    defender_update_count: int
    update_stats: tuple[TD3UpdateStats, ...]


@dataclass(frozen=True)
class PolicyLeaderResult:
    initial_defender_fingerprint: str
    meta_defender_fingerprint: str
    task_updates: tuple[PolicyLeaderTaskUpdate, ...]
    kappa_D: float
    protocol: str = 'policy-td3-leader-update-v1'


class PolicyLeaderTrainer:
    """Estimate gradients at θξ and average their κ_D-scaled deltas at θ."""

    def __init__(
        self,
        *,
        defender_updates: int,
        batch_size: int,
        kappa_D: float,
    ) -> None:
        for value, name in (
            (defender_updates, 'defender_updates'),
            (batch_size, 'batch_size'),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(kappa_D) or kappa_D <= 0:
            raise ValueError('kappa_D must be positive and finite')
        self.defender_updates = defender_updates
        self.batch_size = batch_size
        self.kappa_D = float(kappa_D)

    def train(
        self,
        *,
        defender: TD3Agent,
        tasks: tuple[PolicyLeaderTask, ...],
        collect_fresh,
    ) -> PolicyLeaderResult:
        if defender.role != 'defender':
            raise ValueError('leader policy must have defender role')
        if not tasks:
            raise ValueError('at least one leader task is required')
        initial_fingerprint = defender.fingerprint()
        task_snapshots = []
        task_before_snapshots = []
        task_updates = []
        for task in tasks:
            if task.best_response.role != 'attacker':
                raise ValueError('best response must have attacker role')
            if task.adapted_defender.role != 'defender':
                raise ValueError('adapted policy must have defender role')
            if task.defender_replay.role != 'defender':
                raise ValueError('leader replay must have defender role')
            task_defender = task.adapted_defender.clone()
            task_before = task_defender.snapshot()
            task_defender.set_learning_rate(self.kappa_D)
            response_guard = task.best_response.freeze_guard()
            collect_fresh(
                task.task_id, task_defender, task.best_response,
                task.defender_replay,
            )
            response_guard.verify()
            stats = []
            for _ in range(self.defender_updates):
                stats.append(task_defender.update(
                    task.defender_replay.sample(self.batch_size),
                ))
                response_guard.verify()
            snapshot = task_defender.snapshot()
            task_before_snapshots.append(task_before)
            task_snapshots.append(snapshot)
            task_updates.append(PolicyLeaderTaskUpdate(
                task.task_id,
                response_guard.fingerprint,
                task_defender.fingerprint(),
                len(stats),
                tuple(stats),
            ))
        _apply_average_task_deltas(
            defender, tuple(task_before_snapshots), tuple(task_snapshots),
        )
        return PolicyLeaderResult(
            initial_fingerprint,
            defender.fingerprint(),
            tuple(task_updates),
            self.kappa_D,
        )


_NETWORK_STATES = (
    'actor', 'actor_target', 'critic1', 'critic2',
    'critic1_target', 'critic2_target',
)


def _apply_average_task_deltas(
    meta_agent: TD3Agent,
    before_snapshots,
    after_snapshots,
) -> None:
    if len(before_snapshots) != len(after_snapshots) or not after_snapshots:
        raise ValueError('matching non-empty task snapshots are required')
    meta = meta_agent.snapshot()
    count = float(len(after_snapshots))
    for state_name in _NETWORK_STATES:
        meta_state = getattr(meta, state_name)
        result = {}
        for key, meta_tensor in meta_state.items():
            delta = torch.zeros_like(meta_tensor)
            for before, after in zip(before_snapshots, after_snapshots):
                before_tensor = getattr(before, state_name)[key]
                after_tensor = getattr(after, state_name)[key]
                if before_tensor.shape != meta_tensor.shape or after_tensor.shape != meta_tensor.shape:
                    raise ValueError('task snapshot network structure mismatch')
                delta.add_(after_tensor - before_tensor)
            result[key] = meta_tensor + delta / count
        getattr(meta_agent, state_name).load_state_dict(result)
