"""Structural plug-in contracts for one federated-learning round."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import (
    ClientUpdate,
    RoundRequest,
    RoundState,
    RoundTransition,
)


@runtime_checkable
class ClientSampler(Protocol):
    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]: ...


@runtime_checkable
class LocalTrainer(Protocol):
    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate: ...


@runtime_checkable
class Aggregator(Protocol):
    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState: ...


@runtime_checkable
class ServerOptimizer(Protocol):
    def step(
        self,
        model: ModelState,
        aggregate_delta: ModelState,
        learning_rate: float,
    ) -> ModelState: ...


@runtime_checkable
class RoundExecutor(Protocol):
    def run_round(
        self,
        request: RoundRequest,
        rng: RandomSource,
    ) -> RoundTransition: ...
