"""A Little Is Enough model-poisoning attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights


def craft_alie(
    benign_weights: List[Weights],
    tau: float = 1.5,
) -> Weights:
    """A Little Is Enough attack (Baruch et al., NeurIPS 2019)."""
    if not benign_weights:
        raise ValueError("craft_alie requires at least one benign weight set")
    result = []
    for layer_idx in range(len(benign_weights[0])):
        stack = np.stack([np.asarray(w[layer_idx]) for w in benign_weights]).astype(np.float64)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        result.append((mean - tau * std).astype(np.float32))
    return result


@dataclass
class ALIEAttack(SandboxAttack):
    """A Little Is Enough (ALIE) attack — evades Krum by staying inside benign cluster."""

    tau: float = 1.5
    name: str = "ALIE"
    attack_type: str = "alie"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0 or not ctx.benign_weights:
            return self.fallback_old_weights(ctx)
        crafted = craft_alie(ctx.benign_weights, tau=self.tau)
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]
