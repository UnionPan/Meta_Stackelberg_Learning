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
