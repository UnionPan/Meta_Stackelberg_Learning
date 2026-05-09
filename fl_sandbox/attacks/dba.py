"""Distributed backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader


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
