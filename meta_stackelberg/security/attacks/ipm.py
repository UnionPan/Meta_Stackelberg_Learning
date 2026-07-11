"""Inner Product Manipulation using observed same-round benign updates."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.types import (
    AttackCapabilities,
    RoundAttackContext,
)


class IPMAttack:
    """Submit ``-scale * mean(benign_delta)`` for each malicious slot."""

    capabilities = AttackCapabilities(observes_benign_updates=True)

    def __init__(self, *, scale: float, num_examples_by_client: Mapping[int, int]) -> None:
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise ValueError('IPM scale must be finite and positive')
        values = dict(num_examples_by_client)
        if not values:
            raise ValueError('IPM sample counts must not be empty')
        if any(
            not isinstance(client_id, Integral) or isinstance(client_id, bool)
            for client_id in values
        ):
            raise TypeError('IPM client ids must be non-bool integers')
        if any(client_id < 0 for client_id in values):
            raise ValueError('IPM client ids must be non-negative')
        if any(
            not isinstance(count, Integral) or isinstance(count, bool)
            for count in values.values()
        ):
            raise TypeError('IPM sample counts must be non-bool integers')
        if any(count <= 0 for count in values.values()):
            raise ValueError('IPM sample counts must be positive')
        self.scale = float(scale)
        self.num_examples_by_client = {
            int(client_id): int(count) for client_id, count in values.items()
        }
        self.allowed_client_ids = frozenset(self.num_examples_by_client)

    def craft_round(
        self,
        context: RoundAttackContext,
        rngs: tuple[RandomSource, ...],
    ) -> tuple[ClientUpdate, ...]:
        if not context.benign_updates:
            raise ValueError('IPM requires at least one benign reference update')
        if len(rngs) != len(context.malicious_client_ids):
            raise ValueError('IPM RNG count must match malicious client count')
        missing = set(context.malicious_client_ids) - set(self.num_examples_by_client)
        if missing:
            raise ValueError(f'IPM has no sample count for clients {sorted(missing)}')
        _validate_references(context)
        template_tensors = context.benign_updates[0].delta.tensors
        crafted_tensors = tuple(
            (-self.scale * np.mean(
                np.stack([
                    update.delta.tensors[layer_index]
                    for update in context.benign_updates
                ]),
                axis=0,
            )).astype(template.dtype, copy=False)
            for layer_index, template in enumerate(template_tensors)
        )
        return tuple(
            ClientUpdate(
                client_id=client_id,
                delta=ModelState.from_tensors(crafted_tensors),
                num_examples=self.num_examples_by_client[client_id],
                is_malicious=True,
                metadata={
                    'attack_type': 'ipm',
                    'scale': self.scale,
                    'reference_count': len(context.benign_updates),
                },
            )
            for client_id in context.malicious_client_ids
        )


def _validate_references(context: RoundAttackContext) -> None:
    template = (
        context.global_model
        if context.global_model is not None
        else context.benign_updates[0].delta
    )
    expected_shapes = tuple(tensor.shape for tensor in template.tensors)
    expected_dtypes = tuple(tensor.dtype for tensor in template.tensors)
    for update in context.benign_updates:
        shapes = tuple(tensor.shape for tensor in update.delta.tensors)
        dtypes = tuple(tensor.dtype for tensor in update.delta.tensors)
        if shapes != expected_shapes:
            raise ValueError('benign reference update structure does not match global model')
        if dtypes != expected_dtypes:
            raise ValueError('benign reference update dtype does not match global model')
