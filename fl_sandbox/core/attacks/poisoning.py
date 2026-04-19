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


def craft_ipm(old_weights: Weights, benign_avg_weights: Weights, scale: float = 2.0) -> Weights:
    """Construct an IPM malicious model from the benign average."""
    weight_diff = [old - benign for old, benign in zip(old_weights, benign_avg_weights)]
    crafted_diff = [scale * diff * (-1.0) for diff in weight_diff]
    return [old - diff for old, diff in zip(old_weights, crafted_diff)]


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
        return [crafted] * num_attackers


def _weights_to_vector(weights: Weights) -> np.ndarray:
    return np.concatenate([np.asarray(layer).ravel() for layer in weights], axis=0)


def craft_lmp(
    old_weights: Weights,
    benign_weights_list: List[Weights],
    attacker_local_weights_list: Optional[List[Weights]] = None,
    scale: float = 2.0,
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

    scale: float = 2.0
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
        return [crafted] * num_attackers
