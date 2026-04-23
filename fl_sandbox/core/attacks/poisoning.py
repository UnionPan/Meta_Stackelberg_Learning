"""Model-poisoning attacker implementations and helpers.

Main APIs in this file:
- ``craft_ipm``
- ``craft_lmp``
- ``IPMAttack``
- ``LMPAttack``
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

import numpy as np

from .base import SandboxAttack, Weights, train_on_loader
from fl_sandbox.core.metrics import update_norm


def craft_ipm(old_weights: Weights, benign_avg_weights: Weights, scale: float = 2.0) -> Weights:
    """Construct an IPM malicious model from the benign average.

    Submits old + scale*(old - benign_avg), i.e. the update points in the opposite
    direction from the benign average update, amplified by *scale*.
    """
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


def _weights_to_vector(weights: Weights) -> np.ndarray:
    return np.concatenate([np.asarray(layer).ravel() for layer in weights], axis=0)


def craft_lmp(
    old_weights: Weights,
    benign_weights_list: List[Weights],
    attacker_local_weights_list: Optional[List[Weights]] = None,
    scale: float = 5.0,
) -> Weights:
    """Construct an LMP malicious model aligned with the reference repo's Median_craft_real."""
    temp_weights_lis = list(benign_weights_list) + list(attacker_local_weights_list or [])
    if not temp_weights_lis:
        return old_weights

    from fl_sandbox.core.defender import median_aggregate

    aggregate_weight = median_aggregate(old_weights, temp_weights_lis)
    sign = [np.sign(aggregate - old) for aggregate, old in zip(aggregate_weight, old_weights)]

    max_weight = _weights_to_vector(temp_weights_lis[0]).copy()
    min_weight = _weights_to_vector(temp_weights_lis[0]).copy()
    for weights in temp_weights_lis[1:]:
        vec = _weights_to_vector(weights)
        max_weight = np.maximum(max_weight, vec)
        min_weight = np.minimum(min_weight, vec)

    crafted = []
    count = 0
    for layer in sign:
        new_params = []
        for param in layer.ravel():
            if param == -1.0 and max_weight[count] > 0:
                new_params.append(random.uniform(max_weight[count], scale * max_weight[count]))
            elif param == -1.0 and max_weight[count] <= 0:
                new_params.append(random.uniform(max_weight[count] / scale, max_weight[count]))
            elif param == 1.0 and min_weight[count] > 0:
                new_params.append(random.uniform(min_weight[count] / scale, min_weight[count]))
            elif param == 1.0 and min_weight[count] <= 0:
                new_params.append(random.uniform(scale * min_weight[count], min_weight[count]))
            elif param == 0.0:
                new_params.append(0.0)
            elif np.isnan(param):
                new_params.append(random.uniform(min_weight[count], max_weight[count]))
            else:
                new_params.append(random.uniform(min_weight[count], max_weight[count]))
            count += 1
        crafted.append(np.asarray(new_params, dtype=layer.dtype).reshape(layer.shape))

    return crafted


@dataclass
class LMPAttack(SandboxAttack):
    """Local Model Poisoning against coordinate-wise robust aggregators."""

    scale: float = 5.0
    name: str = "LMP"
    attack_type: str = "lmp"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0:
            return self.fallback_old_weights(ctx)

        attacker_local_weights = []
        for attacker_id in ctx.selected_attacker_ids:
            loader = self.selected_attacker_loader(ctx, attacker_id)
            if loader is None:
                continue
            attacker_local_weights.append(train_on_loader(ctx, loader))

        if not ctx.benign_weights and not attacker_local_weights:
            return self.fallback_old_weights(ctx)

        crafted = craft_lmp(
            ctx.old_weights,
            ctx.benign_weights,
            attacker_local_weights_list=attacker_local_weights,
            scale=self.scale,
        )

        # Prevent NaN divergence under FedAvg: if the crafted update norm is much
        # larger than the mean benign update norm, scale it down while keeping the
        # attack direction.  The ceiling is generous (scale * 20 * mean_benign_norm)
        # so this only kicks in when the feedback loop would otherwise cause overflow.
        if ctx.benign_weights:
            benign_norms = [update_norm(ctx.old_weights, w) for w in ctx.benign_weights]
            mean_benign = float(np.mean(benign_norms)) if benign_norms else 0.0
            if mean_benign > 0:
                crafted_norm = update_norm(ctx.old_weights, crafted)
                max_norm = self.scale * mean_benign * 20.0
                if crafted_norm > max_norm:
                    ratio = max_norm / crafted_norm
                    crafted = [
                        old + ratio * (c - old)
                        for old, c in zip(ctx.old_weights, crafted)
                    ]

        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]
