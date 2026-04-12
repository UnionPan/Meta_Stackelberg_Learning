"""Standalone attacker implementations for isolated validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import List, Optional

import numpy as np
import torch

from .context import RoundContext


Weights = List[np.ndarray]


def craft_ipm(old_weights: Weights, benign_avg_weights: Weights, scaling: float = 2.0) -> Weights:
    """Construct an IPM malicious model from the benign average."""
    weight_diff = [old - benign for old, benign in zip(old_weights, benign_avg_weights)]
    crafted_diff = [scaling * diff * (-1.0) for diff in weight_diff]
    return [old - diff for old, diff in zip(old_weights, crafted_diff)]


def _weights_to_vector(weights: Weights) -> np.ndarray:
    return np.concatenate([np.asarray(layer).ravel() for layer in weights], axis=0)


def craft_lmp(old_weights: Weights, benign_weights_list: List[Weights], scale: float = 2.0) -> Weights:
    """Construct an LMP malicious model against coordinate-wise robust aggregation."""
    if not benign_weights_list:
        return old_weights

    benign_avg = [
        np.mean([weights[layer] for weights in benign_weights_list], axis=0)
        for layer in range(len(old_weights))
    ]
    sign = [np.sign(avg - old) for avg, old in zip(benign_avg, old_weights)]

    max_weight = _weights_to_vector(benign_weights_list[0]).copy()
    min_weight = _weights_to_vector(benign_weights_list[0]).copy()
    for weights in benign_weights_list[1:]:
        vec = _weights_to_vector(weights)
        max_weight = np.maximum(max_weight, vec)
        min_weight = np.minimum(min_weight, vec)

    crafted = []
    count = 0
    for layer in sign:
        new_params = []
        for param in layer.ravel():
            if param == -1.0 and max_weight[count] > 0:
                new_params.append(random.uniform((scale - 1.0) * max_weight[count], scale * max_weight[count]))
            elif param == -1.0:
                new_params.append(random.uniform(max_weight[count] / (scale - 1.0), max_weight[count]) / scale)
            elif param == 1.0 and min_weight[count] > 0:
                new_params.append(random.uniform(min_weight[count] / scale, min_weight[count]) / (scale - 1.0))
            elif param == 1.0:
                new_params.append(random.uniform(scale * min_weight[count], (scale - 1.0) * min_weight[count]))
            elif param == 0.0:
                new_params.append(0.0)
            else:
                new_params.append(random.uniform(min_weight[count], max_weight[count]))
            count += 1
        crafted.append(np.asarray(new_params, dtype=layer.dtype).reshape(layer.shape))

    return crafted


class SandboxAttack:
    """Minimal attacker interface for the sandbox."""

    name: str = "base"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        raise NotImplementedError


def _set_model_weights(model: torch.nn.Module, weights: Weights, device: torch.device) -> None:
    with torch.no_grad():
        for target, source in zip(model.state_dict().values(), weights):
            source_tensor = torch.as_tensor(source, device=device, dtype=target.dtype)
            target.copy_(source_tensor.reshape_as(target))


def _get_model_weights(model: torch.nn.Module) -> Weights:
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def _train_on_loader(ctx: RoundContext, loader) -> Weights:
    if ctx.model is None or loader is None or ctx.device is None:
        return ctx.old_weights

    model = copy.deepcopy(ctx.model).to(ctx.device)
    if getattr(ctx.device, "type", None) == "cuda":
        model = model.to(memory_format=torch.channels_last)
    _set_model_weights(model, ctx.old_weights, ctx.device)
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

    return _get_model_weights(model)


def _bounded_local_lr(action_value: float) -> float:
    return float(action_value) * 0.05 + 0.05


def _bounded_local_epochs(action_value: float) -> int:
    local_epochs = int(float(action_value) * 5 + 6)
    return max(1, min(11, local_epochs))


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


@dataclass
class LMPAttack(SandboxAttack):
    """Local Model Poisoning against coordinate-wise robust aggregators."""

    scale: float = 2.0
    name: str = "LMP"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0 or not ctx.benign_weights:
            return [ctx.old_weights] * num_attackers

        crafted = craft_lmp(ctx.old_weights, ctx.benign_weights, scale=self.scale)
        return [crafted] * num_attackers


@dataclass
class BFLAttack(SandboxAttack):
    """Backdoor FL using a single global trigger pattern."""

    poison_frac: float = 1.0
    name: str = "BFL"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0:
            return []
        poisoned_loader = ctx.global_poisoned_train_loader
        if poisoned_loader is None:
            poisoned_loader = (ctx.poisoned_train_iters or {}).get("global")
        if poisoned_loader is None:
            return [ctx.old_weights] * num_attackers
        return [_train_on_loader(ctx, poisoned_loader) for _ in range(num_attackers)]


@dataclass
class DBAAttack(SandboxAttack):
    """Distributed Backdoor Attack using one sub-trigger per attacker."""

    num_sub_triggers: int = 4
    poison_frac: float = 0.5
    name: str = "DBA"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0:
            return []
        sub_loaders = ctx.sub_trigger_train_loaders
        if sub_loaders is None:
            sub_loaders = (ctx.poisoned_train_iters or {}).get("sub_triggers", [])
        if not sub_loaders:
            return [ctx.old_weights] * num_attackers

        malicious_weights: List[Weights] = []
        max_subs = min(self.num_sub_triggers, len(sub_loaders))
        for attacker_offset in range(num_attackers):
            sub_idx = attacker_offset % max_subs
            malicious_weights.append(_train_on_loader(ctx, sub_loaders[sub_idx]))
        return malicious_weights


@dataclass
class RLAttack(SandboxAttack):
    """Adaptive untargeted model poisoning driven by an attacker action vector."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "RL"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0 or not ctx.benign_weights:
            return [ctx.old_weights] * num_attackers

        action = attacker_action if attacker_action is not None else ctx.attacker_action
        if action is None or ctx.attacker_train_iter is None:
            return [ctx.old_weights] * num_attackers

        scaling = float(action[0]) * 5.0 + 5.0
        local_lr = _bounded_local_lr(float(action[1]))
        local_epochs = _bounded_local_epochs(float(action[2]))
        train_ctx = copy.copy(ctx)
        train_ctx.lr = local_lr
        train_ctx.local_epochs = local_epochs

        attacker_weights = _train_on_loader(train_ctx, ctx.attacker_train_iter)
        crafted = craft_ipm(ctx.old_weights, attacker_weights, scaling=scaling)
        return [crafted] * num_attackers


@dataclass
class BRLAttack(SandboxAttack):
    """Adaptive backdoor attack driven by an attacker action vector."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "BRL"

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0:
            return []

        action = attacker_action if attacker_action is not None else ctx.attacker_action
        if action is None:
            return [ctx.old_weights] * num_attackers

        poisoned_loader = ctx.global_poisoned_train_loader
        if poisoned_loader is None:
            poisoned_loader = (ctx.poisoned_train_iters or {}).get("global")
        if poisoned_loader is None:
            return [ctx.old_weights] * num_attackers

        local_lr = _bounded_local_lr(float(action[1]))
        local_epochs = _bounded_local_epochs(float(action[2]))
        train_ctx = copy.copy(ctx)
        train_ctx.lr = local_lr
        train_ctx.local_epochs = local_epochs
        return [_train_on_loader(train_ctx, poisoned_loader) for _ in range(num_attackers)]
