"""Immutable records crossing federated-learning module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, TypeAlias

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSnapshot


Scalar: TypeAlias = bool | int | float | str


def _frozen_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    delta: ModelState
    num_examples: int
    is_malicious: bool = False
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError('client_id must be non-negative')
        if self.num_examples <= 0:
            raise ValueError('num_examples must be positive')
        object.__setattr__(self, 'metadata', _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RoundState:
    round_index: int
    global_model: ModelState
    random_snapshot: RandomSnapshot
    component_states: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError('round_index must be non-negative')
        object.__setattr__(self, 'component_states', _frozen_mapping(self.component_states))


@dataclass(frozen=True)
class RoundRequest:
    task_id: str
    state: RoundState
    sample_size: int
    server_lr: float = 1.0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id must not be empty')
        if self.sample_size <= 0:
            raise ValueError('sample_size must be positive')
        if not math.isfinite(float(self.server_lr)):
            raise ValueError('server_lr must be finite')


@dataclass(frozen=True)
class RoundTransition:
    task_id: str
    state_before: RoundState
    sampled_clients: tuple[int, ...]
    benign_updates: tuple[ClientUpdate, ...]
    malicious_updates: tuple[ClientUpdate, ...]
    aggregate_delta: ModelState
    state_after: RoundState
    public_signals: Mapping[str, object] = field(default_factory=dict)
    private_diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id must not be empty')
        if len(self.sampled_clients) != len(set(self.sampled_clients)):
            raise ValueError('sampled_clients contains duplicate client ids')
        if self.state_after.round_index != self.state_before.round_index + 1:
            raise ValueError('state_after round_index must advance by exactly one')
        object.__setattr__(self, 'public_signals', _frozen_mapping(self.public_signals))
        object.__setattr__(self, 'private_diagnostics', _frozen_mapping(self.private_diagnostics))
