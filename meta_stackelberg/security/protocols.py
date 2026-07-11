"""Structural plug-in contracts for adversarial federated execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.types import AttackCapabilities, AttackContext


@runtime_checkable
class MaliciousPopulation(Protocol):
    def contains(self, client_id: int) -> bool: ...


@runtime_checkable
class MaliciousUpdateGenerator(Protocol):
    capabilities: AttackCapabilities

    def craft(
        self,
        context: AttackContext,
        rng: RandomSource,
    ) -> ClientUpdate: ...
