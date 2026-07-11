"""Median-craft Local Model Poisoning with explicit replayable RNGs."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks.ipm import _validate_references
from meta_stackelberg.security.types import AttackCapabilities, RoundAttackContext


class LMPAttack:
    """Craft local models outside benign coordinate ranges against Median."""

    capabilities = AttackCapabilities(
        needs_global_model=True,
        observes_benign_updates=True,
    )

    def __init__(self, *, scale: float, num_examples_by_client: Mapping[int, int]) -> None:
        if not math.isfinite(float(scale)) or scale < 1.0:
            raise ValueError('LMP scale must be finite and at least one')
        values = dict(num_examples_by_client)
        if not values:
            raise ValueError('LMP sample counts must not be empty')
        if any(
            not isinstance(client_id, Integral) or isinstance(client_id, bool)
            for client_id in values
        ):
            raise TypeError('LMP client ids must be non-bool integers')
        if any(client_id < 0 for client_id in values):
            raise ValueError('LMP client ids must be non-negative')
        if any(
            not isinstance(count, Integral) or isinstance(count, bool)
            for count in values.values()
        ):
            raise TypeError('LMP sample counts must be non-bool integers')
        if any(count <= 0 for count in values.values()):
            raise ValueError('LMP sample counts must be positive')
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
            raise ValueError('LMP requires at least one benign reference update')
        if context.global_model is None:
            raise ValueError('LMP requires the global model')
        if len(rngs) != len(context.malicious_client_ids):
            raise ValueError('LMP RNG count must match malicious client count')
        missing = set(context.malicious_client_ids) - set(self.num_examples_by_client)
        if missing:
            raise ValueError(f'LMP has no sample count for clients {sorted(missing)}')
        _validate_references(context)
        reference_models = tuple(
            tuple(
                (global_tensor + update.delta.tensors[layer_index]).astype(
                    global_tensor.dtype,
                    copy=False,
                )
                for layer_index, global_tensor in enumerate(context.global_model.tensors)
            )
            for update in context.benign_updates
        )
        updates = []
        for client_id, rng in zip(context.malicious_client_ids, rngs):
            crafted_delta = []
            for layer_index, old_tensor in enumerate(context.global_model.tensors):
                references = np.stack([
                    model[layer_index] for model in reference_models
                ])
                direction = np.sign(np.median(references, axis=0) - old_tensor)
                minimum = np.min(references, axis=0)
                maximum = np.max(references, axis=0)
                crafted_model = np.empty_like(old_tensor)
                for index in np.ndindex(old_tensor.shape):
                    sign = float(direction[index])
                    low = float(minimum[index])
                    high = float(maximum[index])
                    if sign < 0.0 and high > 0.0:
                        value = rng.python.uniform(high, self.scale * high)
                    elif sign < 0.0:
                        value = rng.python.uniform(high / self.scale, high)
                    elif sign > 0.0 and low > 0.0:
                        value = rng.python.uniform(low / self.scale, low)
                    elif sign > 0.0:
                        value = rng.python.uniform(self.scale * low, low)
                    else:
                        value = 0.0
                    crafted_model[index] = value
                crafted_delta.append(
                    (crafted_model - old_tensor).astype(old_tensor.dtype, copy=False)
                )
            updates.append(ClientUpdate(
                client_id=client_id,
                delta=ModelState.from_tensors(crafted_delta),
                num_examples=self.num_examples_by_client[client_id],
                is_malicious=True,
                metadata={
                    'attack_type': 'lmp',
                    'scale': self.scale,
                    'reference_count': len(context.benign_updates),
                    'variant': 'median_craft_real',
                },
            ))
        return tuple(updates)
