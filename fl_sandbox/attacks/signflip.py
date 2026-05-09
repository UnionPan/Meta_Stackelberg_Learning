"""Sign flipping model-poisoning attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights


@dataclass
class SignFlipAttack(SandboxAttack):
    """Sign Flipping attack — negates the mean benign update."""

    name: str = "SignFlip"
    attack_type: str = "signflip"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0 or not ctx.benign_weights:
            return self.fallback_old_weights(ctx)
        benign_avg = [
            np.mean([w[i] for w in ctx.benign_weights], axis=0)
            for i in range(len(ctx.old_weights))
        ]
        crafted = [old + (old - avg) for old, avg in zip(ctx.old_weights, benign_avg)]
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]
