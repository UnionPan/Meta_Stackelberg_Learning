"""Shared construction and validation for trainer-backed attacks."""

from __future__ import annotations

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.protocols import LocalTrainer
from meta_stackelberg.federated.engine.round_kernel import make_client_state
from meta_stackelberg.federated.types import ClientUpdate, RoundState, Scalar
from meta_stackelberg.security.types import AttackCapabilities, AttackContext


LOCAL_MODEL_CAPABILITIES = AttackCapabilities(
    needs_global_model=True,
    needs_local_data=True,
)


def train_base_update(
    trainer: LocalTrainer,
    context: AttackContext,
    rng: RandomSource,
) -> ClientUpdate:
    sanitized_state = make_client_state(
        RoundState(
            round_index=context.round_index,
            global_model=context.global_model,
            random_snapshot=rng.capture(),
            component_states={},
        ),
        rng,
    )
    update = trainer.train(context.client_id, sanitized_state, rng)
    if update.client_id != context.client_id:
        raise ValueError(
            f'base trainer returned update for client {update.client_id}, '
            f'expected client {context.client_id}'
        )
    if update.is_malicious:
        raise ValueError('base trainer returned an already malicious update')
    return update


def as_malicious_update(
    base: ClientUpdate,
    *,
    delta: ModelState | None = None,
    metadata: dict[str, Scalar] | None = None,
) -> ClientUpdate:
    return ClientUpdate(
        client_id=base.client_id,
        delta=base.delta if delta is None else delta,
        num_examples=base.num_examples,
        is_malicious=True,
        metadata=dict(base.metadata) if metadata is None else metadata,
    )
