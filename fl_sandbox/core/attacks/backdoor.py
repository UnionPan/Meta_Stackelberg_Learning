"""Backdoor-style attacker implementations.

Main APIs in this file:
- ``BFLAttack``
- ``DBAAttack``
- ``BRLAttack``
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import List, Optional

import numpy as np

from .base import (
    SandboxAttack,
    Weights,
    bounded_local_epochs,
    bounded_local_lr,
    train_on_loader,
)


@dataclass
class BFLAttack(SandboxAttack):
    """Backdoor FL using a single global trigger pattern."""

    poison_frac: float = 1.0
    name: str = "BFL"
    attack_type: str = "bfl"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []
        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            poisoned_loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if poisoned_loader is None:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            malicious_weights.append(train_on_loader(ctx, poisoned_loader))
        return malicious_weights


@dataclass
class DBAAttack(SandboxAttack):
    """Distributed Backdoor Attack using one sub-trigger per attacker."""

    num_sub_triggers: int = 4
    poison_frac: float = 0.5
    name: str = "DBA"
    attack_type: str = "dba"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []

        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            sub_loaders = self.sub_trigger_loaders_for_attacker(ctx, attacker_id)
            if not sub_loaders:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            max_subs = min(self.num_sub_triggers, len(sub_loaders))
            sub_idx = attacker_id % max_subs
            malicious_weights.append(train_on_loader(ctx, sub_loaders[sub_idx]))
        return malicious_weights


@dataclass
class BRLAttack(SandboxAttack):
    """Adaptive backdoor attack aligned with the reference paper implementation."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "BRL"
    attack_type: str = "brl"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []

        action = self.resolve_action(ctx, attacker_action, default_action=self.default_action)
        if action is None:
            return self.fallback_old_weights(ctx)

        poisoned_loader = self.global_poisoned_loader(ctx)
        if poisoned_loader is None:
            return self.fallback_old_weights(ctx)

        train_ctx = copy.copy(ctx)
        train_ctx.lr = bounded_local_lr(float(action[1]))
        train_ctx.local_epochs = bounded_local_epochs(float(action[2]))
        malicious_weights: List[Weights] = []
        for _ in ctx.selected_attacker_ids:
            malicious_weights.append(train_on_loader(train_ctx, poisoned_loader))
        return malicious_weights
