"""Neutral mechanics shared by clean and adversarial round executors."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from meta_stackelberg.core.model_state import state_l2_norm
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.protocols import Aggregator, ClientSampler, ServerOptimizer
from meta_stackelberg.federated.types import (
    ClientUpdate,
    RoundRequest,
    RoundState,
    RoundTransition,
)


@dataclass(frozen=True)
class ClientSlot:
    client_id: int
    rng: RandomSource


def prepare_client_slots(
    request: RoundRequest,
    parent_rng: RandomSource,
    sampler: ClientSampler,
) -> tuple[ClientSlot, ...]:
    """Restore the parent and derive sampling/per-slot child streams."""

    parent_rng.restore(request.state.random_snapshot)
    sampling_rng = parent_rng.spawn()
    raw_client_ids = tuple(sampler.sample(request, sampling_rng))
    _validate_sample(raw_client_ids, request.sample_size)
    sampled_clients = tuple(int(client_id) for client_id in raw_client_ids)
    return tuple(
        ClientSlot(client_id=client_id, rng=parent_rng.spawn())
        for client_id in sampled_clients
    )


def make_client_state(state: RoundState, client_rng: RandomSource) -> RoundState:
    """Expose only model/round/RNG state to any local client trainer."""

    return RoundState(
        round_index=state.round_index,
        global_model=state.global_model,
        random_snapshot=client_rng.capture(),
        component_states={},
    )


def finalize_round(
    *,
    request: RoundRequest,
    parent_rng: RandomSource,
    sampled_clients: tuple[int, ...],
    ordered_updates: Sequence[ClientUpdate],
    aggregator: Aggregator,
    server_optimizer: ServerOptimizer,
    private_diagnostics: Mapping[str, object] | None = None,
) -> RoundTransition:
    """Aggregate in sampled order and construct the next immutable transition."""

    updates = tuple(ordered_updates)
    _validate_updates(request.state, sampled_clients, updates)
    aggregate_delta = aggregator.aggregate(updates)
    _validate_model_structure(request.state, aggregate_delta, source='aggregate delta')
    _validate_finite(aggregate_delta, source='aggregate delta')
    next_model = server_optimizer.step(
        request.state.global_model,
        aggregate_delta,
        request.server_lr,
    )
    next_state = RoundState(
        round_index=request.state.round_index + 1,
        global_model=next_model,
        random_snapshot=parent_rng.capture(),
        component_states=request.state.component_states,
    )
    benign_updates = tuple(update for update in updates if not update.is_malicious)
    malicious_updates = tuple(update for update in updates if update.is_malicious)
    return RoundTransition(
        task_id=request.task_id,
        state_before=request.state,
        sampled_clients=sampled_clients,
        benign_updates=benign_updates,
        malicious_updates=malicious_updates,
        aggregate_delta=aggregate_delta,
        state_after=next_state,
        public_signals={
            'sampled_client_count': len(sampled_clients),
            'aggregate_norm': state_l2_norm(aggregate_delta),
        },
        private_diagnostics={} if private_diagnostics is None else private_diagnostics,
    )


def _validate_sample(client_ids: tuple[int, ...], expected_size: int) -> None:
    if len(client_ids) != expected_size:
        raise ValueError(f'sampler returned {len(client_ids)} clients, expected {expected_size}')
    if len(client_ids) != len(set(client_ids)):
        raise ValueError('sampler returned duplicate client ids')
    if any(not isinstance(client_id, Integral) or isinstance(client_id, bool) for client_id in client_ids):
        raise ValueError('sampler must return non-bool integers as client ids')
    if any(client_id < 0 for client_id in client_ids):
        raise ValueError('sampler returned a negative client id')


def _validate_updates(
    state: RoundState,
    sampled_clients: tuple[int, ...],
    updates: tuple[ClientUpdate, ...],
) -> None:
    if len(updates) != len(sampled_clients):
        raise ValueError(
            f'update count {len(updates)} does not match sampled count {len(sampled_clients)}'
        )
    for expected_client, update in zip(sampled_clients, updates):
        if update.client_id != expected_client:
            raise ValueError(
                f'update for client {update.client_id} does not match sampled client '
                f'{expected_client}'
            )
        _validate_model_structure(state, update.delta, source=f'client {update.client_id} update')
        _validate_finite(update.delta, source=f'client {update.client_id} update')


def _validate_model_structure(state: RoundState, candidate, *, source: str) -> None:
    expected = tuple(tensor.shape for tensor in state.global_model.tensors)
    actual = tuple(tensor.shape for tensor in candidate.tensors)
    if actual != expected:
        raise ValueError(f'{source} structure {actual} does not match global model {expected}')
    expected_dtypes = tuple(tensor.dtype for tensor in state.global_model.tensors)
    actual_dtypes = tuple(tensor.dtype for tensor in candidate.tensors)
    if actual_dtypes != expected_dtypes:
        raise ValueError(
            f'{source} dtype {actual_dtypes} does not match global model {expected_dtypes}'
        )


def _validate_finite(candidate, *, source: str) -> None:
    if any(not np.all(np.isfinite(tensor)) for tensor in candidate.tensors):
        raise ValueError(f'{source} must contain only finite values')
