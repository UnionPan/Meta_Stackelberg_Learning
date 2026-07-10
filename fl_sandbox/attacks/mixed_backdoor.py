"""Composite backdoor attacker with fixed per-client attack roles."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights
from fl_sandbox.attacks.bfl import BFLAttack
from fl_sandbox.attacks.dba import DBAAttack
from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack


@dataclass
class MixedBackdoorAttack(SandboxAttack):
    """Split malicious clients across BFL, DBA, and RL-backdoor in one FL round."""

    total_attackers: int
    bfl_attack: SandboxAttack = field(default_factory=BFLAttack)
    dba_attack: SandboxAttack = field(default_factory=DBAAttack)
    rl_backdoor_attack: SandboxAttack = field(default_factory=RLBackdoorAttack)
    name: str = "MixedBackdoor"
    attack_type: str = "mixed_backdoor"

    def _roles(self) -> Dict[str, set[int]]:
        attacker_ids = list(range(max(0, int(self.total_attackers))))
        groups = np.array_split(np.asarray(attacker_ids, dtype=int), 3)
        return {
            "bfl": set(int(value) for value in groups[0].tolist()),
            "dba": set(int(value) for value in groups[1].tolist()),
            "rl_backdoor": set(int(value) for value in groups[2].tolist()),
        }

    def observe_round(self, ctx) -> None:
        for _, child, child_ctx in self._iter_children(ctx):
            child.observe_round(child_ctx)

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        weights_by_attacker: Dict[int, Weights] = {}
        for _, child, child_ctx in self._iter_children(ctx):
            child_weights = child.execute(child_ctx, attacker_action=attacker_action)
            for attacker_id, weights in zip(child_ctx.selected_attacker_ids, child_weights):
                weights_by_attacker[int(attacker_id)] = weights
        return [
            weights_by_attacker.get(int(attacker_id), self.clone_old_weights(ctx))
            for attacker_id in ctx.selected_attacker_ids
        ]

    def after_round(self, **kwargs) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for role, child in (
            ("bfl", self.bfl_attack),
            ("dba", self.dba_attack),
            ("rl_backdoor", self.rl_backdoor_attack),
        ):
            after_round = getattr(child, "after_round", None)
            if after_round is None:
                continue
            child_metrics = after_round(**kwargs) or {}
            for key, value in child_metrics.items():
                metrics[f"{role}_{key}"] = float(value)
        return metrics

    def _iter_children(self, ctx):
        roles = self._roles()
        selected = [int(attacker_id) for attacker_id in ctx.selected_attacker_ids]
        for role, child in (
            ("bfl", self.bfl_attack),
            ("dba", self.dba_attack),
            ("rl_backdoor", self.rl_backdoor_attack),
        ):
            child_ids = [attacker_id for attacker_id in selected if attacker_id in roles[role]]
            if not child_ids:
                continue
            child_ctx = copy.copy(ctx)
            child_ctx.selected_attacker_ids = child_ids
            loaders = getattr(ctx, "selected_attacker_train_loaders", None)
            if loaders is not None:
                child_ctx.selected_attacker_train_loaders = {
                    attacker_id: loaders[attacker_id]
                    for attacker_id in child_ids
                    if attacker_id in loaders
                }
            yield role, child, child_ctx
