"""Deterministic orchestration for one clean federated-learning round."""

from __future__ import annotations

from meta_stackelberg.core.model_state import state_l2_norm
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.protocols import Aggregator, ClientSampler, LocalTrainer, ServerOptimizer
from meta_stackelberg.federated.types import RoundRequest, RoundState, RoundTransition


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
        rng.restore(request.state.random_snapshot)
        sampling_rng = rng.spawn()
        sampled_clients = tuple(self.sampler.sample(request, sampling_rng))
        self._validate_sample(sampled_clients, request.sample_size)
        client_rngs = tuple(rng.spawn() for _ in sampled_clients)

        benign_updates = tuple(
            self.trainer.train(client_id, request.state, client_rng)
            for client_id, client_rng in zip(sampled_clients, client_rngs)
        )
        for client_id, update in zip(sampled_clients, benign_updates):
            if update.client_id != client_id:
                raise ValueError(
                    f'trainer returned update for client {update.client_id}, expected {client_id}'
                )
            if update.is_malicious:
                raise ValueError('clean RoundEngine received a malicious client update')

        aggregate_delta = self.aggregator.aggregate(benign_updates)
        next_model = self.server_optimizer.step(
            request.state.global_model,
            aggregate_delta,
            request.server_lr,
        )
        next_state = RoundState(
            round_index=request.state.round_index + 1,
            global_model=next_model,
            random_snapshot=rng.capture(),
            component_states=request.state.component_states,
        )
        return RoundTransition(
            task_id=request.task_id,
            state_before=request.state,
            sampled_clients=sampled_clients,
            benign_updates=benign_updates,
            malicious_updates=(),
            aggregate_delta=aggregate_delta,
            state_after=next_state,
            public_signals={
                'sampled_client_count': len(sampled_clients),
                'aggregate_norm': state_l2_norm(aggregate_delta),
            },
            private_diagnostics={},
        )

    @staticmethod
    def _validate_sample(client_ids: tuple[int, ...], expected_size: int) -> None:
        if len(client_ids) != expected_size:
            raise ValueError(
                f'sampler returned {len(client_ids)} clients, expected {expected_size}'
            )
        if len(client_ids) != len(set(client_ids)):
            raise ValueError('sampler returned duplicate client ids')
        if any(client_id < 0 for client_id in client_ids):
            raise ValueError('sampler returned a negative client id')
