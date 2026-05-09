"""Inner Product Manipulation attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights


def craft_ipm(old_weights: Weights, benign_avg_weights: Weights, scale: float = 2.0) -> Weights:
    """Construct an IPM malicious model from the benign average."""
    return [old + scale * (old - benign) for old, benign in zip(old_weights, benign_avg_weights)]


@dataclass
class IPMAttack(SandboxAttack):
    """Inner Product Manipulation for isolated validation."""

    scale: float = 2.0
    name: str = "IPM"
    attack_type: str = "ipm"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0 or not ctx.benign_weights:
            return self.fallback_old_weights(ctx)

        benign_avg = [
            np.mean([weights[layer] for weights in ctx.benign_weights], axis=0)
            for layer in range(len(ctx.old_weights))
        ]
        crafted = craft_ipm(ctx.old_weights, benign_avg, scale=self.scale)
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]
