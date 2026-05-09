"""Gaussian-noise attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights


@dataclass
class GaussianAttack(SandboxAttack):
    """Gaussian noise attack — submits weights drawn from N(0, sigma)."""

    sigma: float = 0.01
    name: str = "Gaussian"
    attack_type: str = "gaussian"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0:
            return self.fallback_old_weights(ctx)
        rng = np.random.default_rng()
        crafted = [
            rng.normal(0.0, self.sigma, size=layer.shape).astype(np.float32)
            for layer in ctx.old_weights
        ]
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]
