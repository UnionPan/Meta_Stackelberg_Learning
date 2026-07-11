"""Auditable client-scope wrapper for built-in trainer-backed attacks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.protocols import LocalTrainer
from meta_stackelberg.federated.types import ClientUpdate, RoundState


@dataclass(frozen=True, init=False)
class ScopedLocalTrainer:
    """Permit local training calls only for an immutable declared client set."""

    _trainer: LocalTrainer
    allowed_client_ids: frozenset[int]

    def __init__(self, trainer: LocalTrainer, allowed_client_ids: Iterable[int]) -> None:
        values = tuple(allowed_client_ids)
        if any(not isinstance(client_id, Integral) or isinstance(client_id, bool) for client_id in values):
            raise TypeError('allowed client ids must be integers')
        normalized = frozenset(int(client_id) for client_id in values)
        if any(client_id < 0 for client_id in normalized):
            raise ValueError('allowed client ids must be non-negative')
        object.__setattr__(self, '_trainer', trainer)
        object.__setattr__(self, 'allowed_client_ids', normalized)

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        if client_id not in self.allowed_client_ids:
            raise ValueError(f'client {client_id} is outside scoped local data')
        return self._trainer.train(client_id, state, rng)
