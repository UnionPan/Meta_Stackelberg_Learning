"""Client samplers using explicit random sources."""

from __future__ import annotations

from dataclasses import dataclass

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import RoundRequest


@dataclass(frozen=True)
class UniformClientSampler:
    num_clients: int

    def __post_init__(self) -> None:
        if self.num_clients <= 0:
            raise ValueError('num_clients must be positive')

    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        if request.sample_size > self.num_clients:
            raise ValueError('sample_size cannot exceed num_clients')
        selected = rng.numpy.choice(
            self.num_clients,
            size=request.sample_size,
            replace=False,
        )
        return tuple(int(client_id) for client_id in selected)


@dataclass(frozen=True)
class BenignReferenceClientSampler:
    """Uniform subset sampling conditioned on at least one benign client."""

    num_clients: int
    malicious_ids: frozenset[int]

    def __post_init__(self) -> None:
        if self.num_clients <= 0:
            raise ValueError('num_clients must be positive')
        if not self.malicious_ids:
            raise ValueError('malicious_ids must not be empty')
        if any(
            isinstance(client_id, bool)
            or not isinstance(client_id, int)
            or not 0 <= client_id < self.num_clients
            for client_id in self.malicious_ids
        ):
            raise ValueError('malicious_ids must belong to the client population')
        if len(self.malicious_ids) >= self.num_clients:
            raise ValueError('at least one benign client is required')

    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]:
        if request.sample_size > self.num_clients:
            raise ValueError('sample_size cannot exceed num_clients')
        while True:
            selected = tuple(int(client_id) for client_id in rng.numpy.choice(
                self.num_clients,
                size=request.sample_size,
                replace=False,
            ))
            if any(
                client_id not in self.malicious_ids
                for client_id in selected
            ):
                return selected
