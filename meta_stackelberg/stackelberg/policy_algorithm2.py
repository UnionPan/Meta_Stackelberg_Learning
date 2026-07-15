"""Concrete TD3 composition of paper Reptile Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot, TD3UpdateStats
from meta_stackelberg.stackelberg.algorithm2 import reptile_update_td3


@dataclass(frozen=True)
class PolicyAlgorithm2TaskTrace:
    task: object
    response_fingerprint: str
    adapted_defender_fingerprint: str
    update_stats: tuple[TD3UpdateStats, ...]
    adapted_snapshot: TD3Snapshot | None


@dataclass(frozen=True)
class PolicyAlgorithm2IterationTrace:
    meta_iteration: int
    meta_defender_before: str
    meta_defender_after: str
    tasks: tuple[PolicyAlgorithm2TaskTrace, ...]


@dataclass(frozen=True)
class PolicyAlgorithm2Result:
    iterations: tuple[PolicyAlgorithm2IterationTrace, ...]
    protocol: str = 'paper-reptile-algorithm2-policy-v1'


class PolicyMetaSGAlgorithm2:
    """Adapt K isolated θξ copies for l steps, then apply one Reptile update."""

    def __init__(
        self,
        *,
        T: int,
        K: int,
        l: int,
        batch_size: int,
        kappa: float,
        meta_update_step: float,
    ) -> None:
        for value, name in ((T, 'T'), (K, 'K'), (l, 'l'), (batch_size, 'batch_size')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for value, name in ((kappa, 'kappa'), (meta_update_step, 'meta_update_step')):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.T = T
        self.K = K
        self.l = l
        self.batch_size = batch_size
        self.kappa = float(kappa)
        self.meta_update_step = float(meta_update_step)

    def run(
        self,
        *,
        defender: TD3Agent,
        response_policies: Mapping[object, TD3Agent],
        sample_tasks,
        replay_factory,
        collect_task,
        start_iteration: int = 0,
        iteration_callback=None,
    ) -> PolicyAlgorithm2Result:
        if defender.role != 'defender':
            raise ValueError('meta policy must have defender role')
        if (
            isinstance(start_iteration, bool)
            or not isinstance(start_iteration, int)
            or start_iteration < 0
            or start_iteration > self.T
        ):
            raise ValueError('start_iteration must be within [0, T]')
        iterations = []
        for meta_iteration in range(start_iteration, self.T):
            tasks = tuple(sample_tasks(meta_iteration, self.K))
            if len(tasks) != self.K:
                raise ValueError(f'task sampler must return exactly K={self.K} tasks')
            meta_before = defender.fingerprint()
            traces = []
            snapshots = []
            for task in tasks:
                if task not in response_policies:
                    raise KeyError(f'missing response policy for task {task!r}')
                response = response_policies[task]
                if response.role != 'attacker':
                    raise ValueError('task response must have attacker role')
                response_guard = response.freeze_guard()
                adapted = defender.clone()
                adapted.set_learning_rate(self.kappa)
                replay = replay_factory(task, meta_iteration)
                if replay.role != 'defender':
                    raise ValueError('Algorithm 2 replay must have defender role')
                stats = []
                for step in range(self.l):
                    collect_task(
                        task, step, adapted, response, replay, meta_iteration,
                    )
                    response_guard.verify()
                    stats.append(adapted.update(replay.sample(self.batch_size)))
                    response_guard.verify()
                snapshot = adapted.snapshot()
                snapshots.append(snapshot)
                traces.append(PolicyAlgorithm2TaskTrace(
                    task,
                    response_guard.fingerprint,
                    adapted.fingerprint(),
                    tuple(stats),
                    None,
                ))
            reptile_update_td3(
                defender, tuple(snapshots), meta_step=self.meta_update_step,
            )
            trace = PolicyAlgorithm2IterationTrace(
                meta_iteration,
                meta_before,
                defender.fingerprint(),
                tuple(traces),
            )
            iterations.append(trace)
            if iteration_callback is not None:
                iteration_callback(trace, defender, response_policies)
        return PolicyAlgorithm2Result(tuple(iterations))
