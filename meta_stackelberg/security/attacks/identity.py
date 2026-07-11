"""Exact no-op malicious slot used to validate attack orchestration."""

from __future__ import annotations

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks._base import (
    LOCAL_MODEL_CAPABILITIES,
    as_malicious_update,
    train_base_update,
)
from meta_stackelberg.security.types import AttackContext
from meta_stackelberg.security.training import ScopedLocalTrainer


class IdentityMaliciousUpdateGenerator:
    capabilities = LOCAL_MODEL_CAPABILITIES

    def __init__(self, trainer: ScopedLocalTrainer) -> None:
        if not isinstance(trainer, ScopedLocalTrainer):
            raise TypeError('identity attack requires ScopedLocalTrainer')
        self.trainer = trainer

    @property
    def allowed_client_ids(self) -> frozenset[int]:
        return self.trainer.allowed_client_ids

    def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
        return as_malicious_update(train_base_update(self.trainer, context, rng))
