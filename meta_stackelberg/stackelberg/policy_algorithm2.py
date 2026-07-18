"""Concrete TD3 composition of paper Reptile Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
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
        parallel_tasks: int = 1,
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
        if (
            isinstance(parallel_tasks, bool)
            or not isinstance(parallel_tasks, int)
            or parallel_tasks <= 0
        ):
            raise ValueError('parallel_tasks must be a positive integer')
        self.parallel_tasks = min(parallel_tasks, K)

    def run(
        self,
        *,
        defender: TD3Agent,
        response_policies: Mapping[object, TD3Agent],
        sample_tasks,
        replay_factory,
        collect_task,
        prepare_task=None,
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
            jobs = []
            for task_index, task in enumerate(tasks):
                if task not in response_policies:
                    raise KeyError(f'missing response policy for task {task!r}')
                response = (
                    response_policies[task].clone()
                    if self.parallel_tasks > 1
                    else response_policies[task]
                )
                if response.role != 'attacker':
                    raise ValueError('task response must have attacker role')
                adapted = defender.clone()
                adapted.set_learning_rate(self.kappa)
                if prepare_task is None:
                    replay = replay_factory(task, meta_iteration)
                    collector = (
                        lambda step, current, frozen, target, *,
                        _task=task, _iteration=meta_iteration: collect_task(
                            _task, step, current, frozen, target, _iteration,
                        )
                    )
                else:
                    replay, collector = prepare_task(
                        task, meta_iteration, task_index,
                    )
                if replay.role != 'defender':
                    raise ValueError('Algorithm 2 replay must have defender role')
                jobs.append((task, response, adapted, replay, collector))

            results = self._adapt_jobs(tuple(jobs))
            traces = [result[0] for result in results]
            snapshots = [result[1] for result in results]
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

    def _adapt_jobs(self, jobs):
        guards = tuple(job[1].freeze_guard() for job in jobs)
        stats = [[] for _ in jobs]
        executor = (
            ThreadPoolExecutor(
                max_workers=self.parallel_tasks,
                thread_name_prefix='algorithm2-task',
            )
            if self.parallel_tasks > 1
            else None
        )
        try:
            for step in range(self.l):
                if executor is None:
                    for _, response, adapted, replay, collector in jobs:
                        collector(step, adapted, response, replay)
                else:
                    futures = tuple(
                        executor.submit(
                            collector, step, adapted, response, replay,
                        )
                        for _, response, adapted, replay, collector in jobs
                    )
                    for future in futures:
                        future.result()
                for guard in guards:
                    guard.verify()
                # Keep TD3 optimizer updates in task order.  The expensive FL
                # trajectories run concurrently, while single-GPU optimizer
                # kernels retain serial numerical semantics.
                for index, (_, _, adapted, replay, _) in enumerate(jobs):
                    stats[index].append(adapted.update(replay.sample(
                        self.batch_size,
                        replace=self.batch_size > len(replay),
                    )))
                for guard in guards:
                    guard.verify()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        results = []
        for index, (task, _, adapted, _, _) in enumerate(jobs):
            snapshot = adapted.snapshot()
            results.append((PolicyAlgorithm2TaskTrace(
                task,
                guards[index].fingerprint,
                adapted.fingerprint(),
                tuple(stats[index]),
                None,
            ), snapshot))
        return tuple(results)
