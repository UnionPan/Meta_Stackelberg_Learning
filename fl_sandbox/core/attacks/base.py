"""Base classes and shared helpers for sandbox attackers.

Main APIs in this file:
- ``SandboxAttack``
- ``set_model_weights``
- ``get_model_weights``
- ``train_on_loader``
- ``bounded_local_lr``
- ``bounded_local_epochs``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight environments
    torch = None

from fl_sandbox.core.runtime import RoundContext, Weights


class SandboxAttack(ABC):
    """Shared base class for all sandbox attackers."""

    name: str = "base"
    attack_type: str = "base"

    def observe_round(self, ctx: RoundContext) -> None:
        return None

    def selected_attacker_count(self, ctx: RoundContext) -> int:
        return len(ctx.selected_attacker_ids)

    def clone_old_weights(self, ctx: RoundContext) -> Weights:
        return [layer.copy() for layer in ctx.old_weights]

    def fallback_old_weights(self, ctx: RoundContext) -> List[Weights]:
        return [self.clone_old_weights(ctx) for _ in range(self.selected_attacker_count(ctx))]

    def resolve_action(
        self,
        ctx: RoundContext,
        attacker_action: Optional[np.ndarray],
        *,
        default_action: Optional[tuple[float, ...]] = None,
    ) -> Optional[np.ndarray]:
        action = attacker_action if attacker_action is not None else ctx.attacker_action
        if action is None and default_action is not None:
            action = np.asarray(default_action, dtype=float)
        if action is None:
            return None
        return np.asarray(action, dtype=float)

    def global_poisoned_loader(self, ctx: RoundContext):
        loader = getattr(ctx, "global_poisoned_train_loader", None)
        if loader is not None:
            return loader
        return (getattr(ctx, "poisoned_train_iters", None) or {}).get("global")

    def sub_trigger_loaders(self, ctx: RoundContext):
        loaders = getattr(ctx, "sub_trigger_train_loaders", None)
        if loaders is not None:
            return loaders
        return (getattr(ctx, "poisoned_train_iters", None) or {}).get("sub_triggers", [])

    def selected_attacker_loader(self, ctx: RoundContext, attacker_id: int):
        return (getattr(ctx, "selected_attacker_train_loaders", None) or {}).get(attacker_id)

    def poisoned_train_iters(self, ctx: RoundContext) -> dict[str, Any]:
        return getattr(ctx, "poisoned_train_iters", None) or {}

    def global_poisoned_loader_for_attacker(self, ctx: RoundContext, attacker_id: int):
        return self.poisoned_train_iters(ctx).get("global_by_attacker", {}).get(attacker_id)

    def sub_trigger_loaders_for_attacker(self, ctx: RoundContext, attacker_id: int):
        return self.poisoned_train_iters(ctx).get("sub_triggers_by_attacker", {}).get(attacker_id, [])

    @abstractmethod
    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        """Generate malicious client weights for one FL round."""


def set_model_weights(model, weights: Weights, device) -> None:
    if torch is None:
        raise RuntimeError("torch is required to set model weights for sandbox attackers")
    with torch.no_grad():
        for target, source in zip(model.state_dict().values(), weights):
            source_tensor = torch.as_tensor(source, device=device, dtype=target.dtype)
            target.copy_(source_tensor.reshape_as(target))


def get_model_weights(model) -> Weights:
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def train_on_loader(ctx: RoundContext, loader) -> Weights:
    if torch is None:
        raise RuntimeError("torch is required to train attacker models in the sandbox")
    if ctx.model is None or loader is None or ctx.device is None:
        return ctx.old_weights

    model = copy.deepcopy(ctx.model).to(ctx.device)
    if getattr(ctx.device, "type", None) == "cuda":
        model = model.to(memory_format=torch.channels_last)
    set_model_weights(model, ctx.old_weights, ctx.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ctx.lr)
    use_amp = getattr(ctx.device, "type", None) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for _ in range(max(1, ctx.local_epochs)):
        for images, labels in loader:
            images = images.to(ctx.device, non_blocking=use_amp)
            labels = labels.to(ctx.device, non_blocking=use_amp)
            if use_amp:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=True):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    return get_model_weights(model)


def bounded_local_lr(action_value: float) -> float:
    clipped = float(np.clip(float(action_value), -1.0, 1.0))
    return max(0.0, min(0.1, clipped * 0.05 + 0.05))


def bounded_local_epochs(action_value: float) -> int:
    clipped = float(np.clip(float(action_value), -1.0, 1.0))
    local_epochs = int(clipped * 5 + 6)
    return max(1, min(11, local_epochs))
