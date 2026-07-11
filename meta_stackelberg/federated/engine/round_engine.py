"""Deterministic orchestration for one clean federated-learning round."""

from __future__ import annotations

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.engine.round_kernel import (
    finalize_round,
    make_client_state,
    prepare_client_slots,
)
from meta_stackelberg.federated.protocols import Aggregator, ClientSampler, LocalTrainer, ServerOptimizer
from meta_stackelberg.federated.types import RoundRequest, RoundTransition


class RoundEngine:
    """Compose replaceable round components without owning their algorithms."""

    def __init__(
        self,
        *,
        sampler: ClientSampler,
        trainer: LocalTrainer,
        aggregator: Aggregator,
        server_optimizer: ServerOptimizer,
    ) -> None:
        self.sampler = sampler
        self.trainer = trainer
        self.aggregator = aggregator
        self.server_optimizer = server_optimizer

    def run_round(self, request: RoundRequest, rng: RandomSource) -> RoundTransition:
        slots = prepare_client_slots(request, rng, self.sampler)
        benign_updates = tuple(
            self.trainer.train(
                slot.client_id,
                make_client_state(request.state, slot.rng),
                slot.rng,
            )
            for slot in slots
        )
        for update in benign_updates:
            if update.is_malicious:
                raise ValueError('clean RoundEngine received a malicious client update')
        return finalize_round(
            request=request,
            parent_rng=rng,
            sampled_clients=tuple(slot.client_id for slot in slots),
            ordered_updates=benign_updates,
            aggregator=self.aggregator,
            server_optimizer=self.server_optimizer,
        )
