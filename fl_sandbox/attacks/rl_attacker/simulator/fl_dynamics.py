"""Low-level FL weight utilities shared by the paper RL simulator and attack."""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from fl_sandbox.runtime import Weights


def capture_weights(model: nn.Module) -> Weights:
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def load_weights(model: nn.Module, weights: Weights, device: torch.device) -> None:
    with torch.no_grad():
        for target, source in zip(model.state_dict().values(), weights):
            tensor = torch.as_tensor(source, dtype=target.dtype, device=device)
            target.copy_(tensor.reshape_as(target))


def build_model_from_template(template: nn.Module, weights: Weights, device: torch.device) -> nn.Module:
    model = copy.deepcopy(template).to(device)
    load_weights(model, weights, device)
    return model


def vectorize_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors]) if tensors else torch.zeros(1)


def update_norm(old_weights: Weights, new_weights: Weights) -> float:
    return float(np.sqrt(sum(float(np.sum((new - old) ** 2)) for old, new in zip(old_weights, new_weights))))


def craft_paper_reversal_update(old_weights: Weights, trained_weights: Weights, *, gamma_scale: float) -> Weights:
    """Original clipped-median attacker craft: old + gamma * (old - trained)."""

    return [
        old + float(gamma_scale) * (old - trained)
        for old, trained in zip(old_weights, trained_weights)
    ]
