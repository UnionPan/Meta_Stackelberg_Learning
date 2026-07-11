"""Budgeted reversal of a malicious client's own honest local update."""

from __future__ import annotations

import math

import numpy as np

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks._base import (
    LOCAL_MODEL_CAPABILITIES,
    as_malicious_update,
    train_base_update,
)
from meta_stackelberg.security.types import AttackContext
from meta_stackelberg.security.training import ScopedLocalTrainer


class DeltaReversalAttack:
    capabilities = LOCAL_MODEL_CAPABILITIES

    def __init__(self, *, trainer: ScopedLocalTrainer, budget: float) -> None:
        if not isinstance(trainer, ScopedLocalTrainer):
            raise TypeError('delta reversal attack requires ScopedLocalTrainer')
        if not math.isfinite(float(budget)) or budget < 0.0:
            raise ValueError('budget must be finite and non-negative')
        self.trainer = trainer
        self.budget = float(budget)

    @property
    def allowed_client_ids(self) -> frozenset[int]:
        return self.trainer.allowed_client_ids

    def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
        base = train_base_update(self.trainer, context, rng)
        factor = 1.0 - self.budget
        crafted = ModelState.from_tensors(
            (tensor * factor).astype(tensor.dtype, copy=False)
            for tensor in base.delta.tensors
        )
        base_vector = base.delta.vector().astype(np.float64, copy=False)
        denominator = float(np.linalg.norm(base_vector))
        ratio = 0.0
        if denominator > 0.0:
            displacement = crafted.vector().astype(np.float64, copy=False) - base_vector
            ratio = float(np.linalg.norm(displacement) / denominator)
        metadata = dict(base.metadata)
        metadata.update({
            'attack_type': 'delta_reversal',
            'reversal_budget': self.budget,
            'displacement_ratio': ratio,
        })
        return as_malicious_update(base, delta=crafted, metadata=metadata)
