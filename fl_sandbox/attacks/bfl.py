"""Fixed global-trigger backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader


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
