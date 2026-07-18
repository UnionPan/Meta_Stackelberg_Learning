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
        # The coordinate statistics depend only on the benign references, not
        # on the malicious client.  Compute them once per layer and keep the
        # two interval endpoints needed to sample every crafted local model.
        intervals = []
        for layer_index, old_tensor in enumerate(context.global_model.tensors):
            references = np.stack([
                old_tensor + update.delta.tensors[layer_index]
                for update in context.benign_updates
            ])
            direction = np.sign(np.median(references, axis=0) - old_tensor)
            minimum = np.min(references, axis=0)
            maximum = np.max(references, axis=0)
            negative = direction < 0.0
            positive = direction > 0.0
            first = np.zeros_like(old_tensor)
            second = np.zeros_like(old_tensor)
            negative_positive_range = negative & (maximum > 0.0)
            negative_nonpositive_range = negative & ~negative_positive_range
            positive_positive_range = positive & (minimum > 0.0)
            positive_nonpositive_range = positive & ~positive_positive_range
            first[negative_positive_range] = maximum[negative_positive_range]
            second[negative_positive_range] = (
                self.scale * maximum[negative_positive_range]
            )
            first[negative_nonpositive_range] = (
                maximum[negative_nonpositive_range] / self.scale
            )
            second[negative_nonpositive_range] = maximum[
                negative_nonpositive_range
            ]
            first[positive_positive_range] = (
                minimum[positive_positive_range] / self.scale
            )
            second[positive_positive_range] = minimum[positive_positive_range]
            first[positive_nonpositive_range] = (
                self.scale * minimum[positive_nonpositive_range]
            )
            second[positive_nonpositive_range] = minimum[
                positive_nonpositive_range
            ]
            intervals.append((old_tensor, first, second, direction != 0.0))
        updates = []
        for client_id, rng in zip(context.malicious_client_ids, rngs):
            crafted_delta = []
            for old_tensor, first, second, active in intervals:
                draws = rng.numpy.random(old_tensor.shape)
                crafted_model = (
                    first + (second - first) * draws
                ).astype(old_tensor.dtype, copy=False)
                crafted_model[~active] = 0.0
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
