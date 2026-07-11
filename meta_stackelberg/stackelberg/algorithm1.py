"""Trace-verifiable orchestration of Meta-SG Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Algorithm1Event:
    kind: str
    leader_iteration: int
    task: object | None = None
    step: int | None = None


@dataclass(frozen=True)
class Algorithm1TaskTrace:
    task: object
    adapted_defender: object
    attacker_steps: tuple[object, ...]
    approximate_best_response: object
    defender_gradient: object


@dataclass(frozen=True)
class Algorithm1IterationTrace:
    leader_iteration: int
    tasks: tuple[Algorithm1TaskTrace, ...]
    gradients: tuple[object, ...]


@dataclass(frozen=True)
class Algorithm1Result:
    iterations: tuple[Algorithm1IterationTrace, ...]
    events: tuple[Algorithm1Event, ...]


class MetaSGAlgorithm1:
    def __init__(self, *, N_D: int, K: int, N_A: int) -> None:
        for value, name in ((N_D, 'N_D'), (K, 'K'), (N_A, 'N_A')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.N_D = N_D
        self.K = K
        self.N_A = N_A

    def run(
        self,
        *,
        sample_tasks,
        current_defender,
        adapt_defender,
        update_attacker,
        estimate_defender_gradient,
        apply_leader_update,
    ) -> Algorithm1Result:
        events = []
        iterations = []
        for leader_iteration in range(self.N_D):
            meta_defender = current_defender(leader_iteration)
            tasks = tuple(sample_tasks(leader_iteration, self.K))
            events.append(Algorithm1Event('sample_tasks', leader_iteration))
            if len(tasks) != self.K:
                raise ValueError(f'task sampler must return exactly K={self.K} tasks')
            task_traces = []
            gradients = []
            for task in tasks:
                adapted = adapt_defender(task, meta_defender, leader_iteration)
                events.append(Algorithm1Event('adapt_defender', leader_iteration, task))
                response_steps = []
                for response_step in range(self.N_A):
                    response_steps.append(update_attacker(
                        task, adapted, response_step,
                    ))
                    events.append(Algorithm1Event(
                        'attacker_update', leader_iteration, task, response_step,
                    ))
                approximate_response = response_steps[-1]
                gradient = estimate_defender_gradient(
                    task, adapted, approximate_response,
                )
                gradients.append(gradient)
                events.append(Algorithm1Event('defender_gradient', leader_iteration, task))
                task_traces.append(Algorithm1TaskTrace(
                    task,
                    adapted,
                    tuple(response_steps),
                    approximate_response,
                    gradient,
                ))
            apply_leader_update(leader_iteration, tuple(gradients))
            events.append(Algorithm1Event('leader_update', leader_iteration))
            iterations.append(Algorithm1IterationTrace(
                leader_iteration,
                tuple(task_traces),
                tuple(gradients),
            ))
        return Algorithm1Result(tuple(iterations), tuple(events))
