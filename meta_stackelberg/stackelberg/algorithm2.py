"""Trace-verifiable Reptile meta-learning from Meta-SG Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot


@dataclass(frozen=True)
class Algorithm2Event:
    kind: str
    meta_iteration: int
    task: object | None = None
    step: int | None = None


@dataclass(frozen=True)
class Algorithm2TaskTrace:
    task: object
    adapted_policy: object
    adaptation_steps: tuple[object, ...]


@dataclass(frozen=True)
class Algorithm2IterationTrace:
    meta_iteration: int
    tasks: tuple[Algorithm2TaskTrace, ...]
    meta_update: object


@dataclass(frozen=True)
class Algorithm2Result:
    iterations: tuple[Algorithm2IterationTrace, ...]
    events: tuple[Algorithm2Event, ...]


class MetaSGAlgorithm2:
    """Execute exactly ``T x K x l`` paper Algorithm 2 adaptation calls."""

    def __init__(self, *, T: int, K: int, l: int, meta_step: float) -> None:
        for value, name in ((T, 'T'), (K, 'K'), (l, 'l')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(meta_step) or meta_step <= 0:
            raise ValueError('meta_step must be positive and finite')
        self.T = T
        self.K = K
        self.l = l
        self.meta_step = float(meta_step)

    def run(
        self,
        *,
        sample_tasks,
        clone_for_task,
        adapt_task,
        apply_meta_update,
    ) -> Algorithm2Result:
        events = []
        iterations = []
        for meta_iteration in range(self.T):
            tasks = tuple(sample_tasks(meta_iteration, self.K))
            events.append(Algorithm2Event('sample_tasks', meta_iteration))
            if len(tasks) != self.K:
                raise ValueError(f'task sampler must return exactly K={self.K} tasks')
            adapted_policies = []
            task_traces = []
            for task in tasks:
                adapted_policy = clone_for_task(task)
                events.append(Algorithm2Event('clone_task', meta_iteration, task))
                adaptation_steps = []
                for step in range(self.l):
                    adaptation_steps.append(adapt_task(task, adapted_policy, step))
                    events.append(Algorithm2Event(
                        'adapt_task', meta_iteration, task, step,
                    ))
                adapted_policies.append(adapted_policy)
                task_traces.append(Algorithm2TaskTrace(
                    task, adapted_policy, tuple(adaptation_steps),
                ))
            meta_update = apply_meta_update(
                meta_iteration, tuple(adapted_policies), self.meta_step,
            )
            events.append(Algorithm2Event('meta_update', meta_iteration))
            iterations.append(Algorithm2IterationTrace(
                meta_iteration, tuple(task_traces), meta_update,
            ))
        return Algorithm2Result(tuple(iterations), tuple(events))


_ADAPTABLE_STATES = (
    'actor',
    'actor_target',
    'critic1',
    'critic2',
    'critic1_target',
    'critic2_target',
)


def reptile_update_td3(
    meta_agent: TD3Agent,
    task_snapshots: tuple[TD3Snapshot, ...],
    *,
    meta_step: float,
) -> None:
    """Apply θ ← θ + meta_step/K Σ(θ_task − θ) to all TD3 networks.

    Optimizer state, counters and RNG state are deliberately not averaged: they
    are training-process state rather than the paper's adaptable parameters θ.
    """
    if not task_snapshots:
        raise ValueError('at least one task snapshot is required')
    if not math.isfinite(meta_step) or meta_step <= 0:
        raise ValueError('meta_step must be positive and finite')
    if any(snapshot.role != meta_agent.role for snapshot in task_snapshots):
        raise ValueError('task snapshot role mismatch')
    base = meta_agent.snapshot()
    count = float(len(task_snapshots))
    for state_name in _ADAPTABLE_STATES:
        base_state = getattr(base, state_name)
        updated_state = {}
        for key, base_tensor in base_state.items():
            task_tensors = []
            for snapshot in task_snapshots:
                state = getattr(snapshot, state_name)
                if state.keys() != base_state.keys() or state[key].shape != base_tensor.shape:
                    raise ValueError('task snapshot network structure mismatch')
                task_tensors.append(state[key].to(dtype=base_tensor.dtype))
            delta = sum((tensor - base_tensor for tensor in task_tensors),
                        torch.zeros_like(base_tensor))
            updated_state[key] = base_tensor + (float(meta_step) / count) * delta
        getattr(meta_agent, state_name).load_state_dict(updated_state)
