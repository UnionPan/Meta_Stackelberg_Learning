"""Standalone attacker implementations for isolated validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .context import RoundContext


Weights = List[np.ndarray]


def craft_ipm(old_weights: Weights, benign_avg_weights: Weights, scaling: float = 2.0) -> Weights:
    """Construct an IPM malicious model from the benign average."""
    weight_diff = [old - benign for old, benign in zip(old_weights, benign_avg_weights)]
    crafted_diff = [scaling * diff * (-1.0) for diff in weight_diff]
    return [old - diff for old, diff in zip(old_weights, crafted_diff)]


class SandboxAttack:
    """Minimal attacker interface for the sandbox."""

    name: str = "base"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        raise NotImplementedError


@dataclass
class IPMAttack(SandboxAttack):
    """Inner Product Manipulation for isolated validation."""

    scaling: float = 2.0
    name: str = "IPM"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0 or not ctx.benign_weights:
            return [ctx.old_weights] * num_attackers

        benign_avg = [
            np.mean([weights[layer] for weights in ctx.benign_weights], axis=0)
            for layer in range(len(ctx.old_weights))
        ]
        crafted = craft_ipm(ctx.old_weights, benign_avg, scaling=self.scaling)
        return [crafted] * num_attackers
