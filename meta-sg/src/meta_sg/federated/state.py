"""Design: snapshot and restore parameters in model.parameters() order.
Data flow: model -> detached parameter snapshot; global and local snapshots ->
local parameter delta. These tuples contain parameters, not state_dict buffers.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


ModelState = tuple[torch.Tensor, ...]


def flatten_update(update: ModelState) -> torch.Tensor:
    return torch.cat([part.reshape(-1) for part in update])


def stack_updates(updates: Sequence[ModelState]) -> torch.Tensor:
    return torch.stack([flatten_update(update) for update in updates])


def unflatten_update(vector: torch.Tensor, reference: ModelState) -> ModelState:
    parts = vector.split([part.numel() for part in reference])
    return tuple(part.reshape_as(tensor) for part, tensor in zip(parts, reference, strict=True))


def model_state(model: nn.Module) -> ModelState:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


@torch.no_grad()
def load_model_state(model: nn.Module, state: ModelState) -> None:
    for parameter, value in zip(model.parameters(), state, strict=True):
        parameter.copy_(value)


def state_update(before: ModelState, after: ModelState) -> ModelState:
    return tuple(right - left.to(right) for left, right in zip(before, after, strict=True))


@torch.no_grad()
def apply_model_update(model: nn.Module, update: ModelState) -> None:
    """Add an already aggregated delta without averaging or normalizing it again."""
    for parameter, delta in zip(model.parameters(), update, strict=True):
        parameter.add_(delta.to(parameter))
