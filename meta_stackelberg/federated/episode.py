"""Pure multi-round orchestration for one fixed federated task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.protocols import RoundExecutor
from meta_stackelberg.federated.types import RoundRequest, RoundState, RoundTransition

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    horizon: int
    sample_size: int
    server_lr: float
    initial_state: RoundState

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id must not be empty')
        if self.horizon <= 0:
            raise ValueError('horizon must be positive')
        if self.sample_size <= 0:
            raise ValueError('sample_size must be positive')
        if not math.isfinite(float(self.server_lr)):
            raise ValueError('server_lr must be finite')


@dataclass(frozen=True)
class FederatedTrajectory:
    task_id: str
    initial_state: RoundState
    transitions: tuple[RoundTransition, ...]
    final_state: RoundState

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id must not be empty')
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError('trajectory must contain at least one transition')
        expected_state = self.initial_state
        for transition in transitions:
            _validate_transition(self.task_id, expected_state, transition)
            expected_state = transition.state_after
        if not _round_states_equal(self.final_state, expected_state):
            raise ValueError('final_state does not match the last transition')
        object.__setattr__(self, 'transitions', transitions)


class EpisodeRunner:
    """Run exactly one fixed-task horizon without reward or agent updates."""

    def __init__(self, executor: RoundExecutor) -> None:
        self.executor = executor

    def run(self, spec: EpisodeSpec, rng: RandomSource) -> FederatedTrajectory:
        state = spec.initial_state
        transitions: list[RoundTransition] = []
        for _ in range(spec.horizon):
            transition = self.executor.run_round(
                RoundRequest(
                    task_id=spec.task_id,
                    state=state,
                    sample_size=spec.sample_size,
                    server_lr=spec.server_lr,
                ),
                rng,
            )
            _validate_transition(spec.task_id, state, transition)
            transitions.append(transition)
            state = transition.state_after
        return FederatedTrajectory(
            task_id=spec.task_id,
            initial_state=spec.initial_state,
            transitions=tuple(transitions),
            final_state=state,
        )


def _validate_transition(
    task_id: str,
    expected_state: RoundState,
    transition: RoundTransition,
) -> None:
    if transition.task_id != task_id:
        raise ValueError(
            f'round executor returned task {transition.task_id!r}, expected {task_id!r}'
        )
    if not _round_states_equal(transition.state_before, expected_state):
        raise ValueError('round executor returned a transition from an unrelated state')


def _round_states_equal(left: RoundState, right: RoundState) -> bool:
    if left.round_index != right.round_index:
        return False
    if len(left.global_model.tensors) != len(right.global_model.tensors):
        return False
    if not all(
        np.array_equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(
            left.global_model.tensors,
            right.global_model.tensors,
        )
    ):
        return False
    if not _snapshots_equal(left.random_snapshot, right.random_snapshot):
        return False
    return _nested_equal(dict(left.component_states), dict(right.component_states))


def _snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> bool:
    if left.python_state != right.python_state:
        return False
    if not _nested_equal(left.numpy_state, right.numpy_state):
        return False
    if left.torch_cpu_state is None or right.torch_cpu_state is None:
        return left.torch_cpu_state is None and right.torch_cpu_state is None
    if torch is None:
        return False
    return bool(torch.equal(left.torch_cpu_state, right.torch_cpu_state))


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and bool(
            np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_nested_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)
