"""Trainer-backed targeted backdoor update generation."""

from __future__ import annotations

import math

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks._base import (
    LOCAL_MODEL_CAPABILITIES,
    as_malicious_update,
    train_base_update,
)
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackContext
from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset


class BackdoorLocalUpdateGenerator:
    """Mark an update produced by a poisoned malicious-client dataset as BFL."""

    capabilities = LOCAL_MODEL_CAPABILITIES

    def __init__(
        self,
        *,
        trainer: ScopedLocalTrainer,
        source_class: int,
        target_class: int,
        poison_fraction: float,
    ) -> None:
        if not isinstance(trainer, ScopedLocalTrainer):
            raise TypeError('backdoor attack requires ScopedLocalTrainer')
        normalized_source = class_id(source_class, name='source_class')
        normalized_target = class_id(target_class, name='target_class')
        if normalized_source == normalized_target:
            raise ValueError('source_class and target_class must differ')
        if not math.isfinite(float(poison_fraction)) or not 0.0 <= poison_fraction <= 1.0:
            raise ValueError('poison_fraction must be finite and in [0, 1]')
        self.trainer = trainer
        for client_id in trainer.allowed_client_ids:
            dataset = trainer.dataset_for(client_id)
            if not isinstance(dataset, SourceTargetPoisonedDataset):
                raise ValueError(
                    f'client {client_id} requires TorchLocalTrainer with a poisoned dataset'
                )
            if (
                dataset.source_class != normalized_source
                or dataset.target_class != normalized_target
                or dataset.poison_fraction != float(poison_fraction)
            ):
                raise ValueError(f'client {client_id} poisoned dataset configuration mismatch')
        self.source_class = normalized_source
        self.target_class = normalized_target
        self.poison_fraction = float(poison_fraction)

    @property
    def allowed_client_ids(self) -> frozenset[int]:
        return self.trainer.allowed_client_ids

    def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
        base = train_base_update(self.trainer, context, rng)
        metadata = dict(base.metadata)
        metadata.update({
            'attack_type': 'bfl_backdoor',
            'source_class': self.source_class,
            'target_class': self.target_class,
            'poison_fraction': self.poison_fraction,
        })
        return as_malicious_update(base, metadata=metadata)
