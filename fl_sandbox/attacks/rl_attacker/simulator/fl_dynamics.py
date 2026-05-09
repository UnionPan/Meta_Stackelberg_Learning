"""One-step FL dynamics used by the RL attacker simulator and live attack."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_sandbox.attacks.rl_attacker.action_decoder import AttackParameters
from fl_sandbox.core.runtime import Weights


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


def tail_named_parameters(model: nn.Module, num_tail_layers: int) -> list[tuple[str, nn.Parameter]]:
    params = list(model.named_parameters())
    return params[-max(1, int(num_tail_layers)) :]


def vectorize_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors]) if tensors else torch.zeros(1)


def update_norm(old_weights: Weights, new_weights: Weights) -> float:
    return float(np.sqrt(sum(float(np.sum((new - old) ** 2)) for old, new in zip(old_weights, new_weights))))


def match_update_norm(old_weights: Weights, new_weights: Weights, *, target_norm: float) -> Weights:
    current_norm = update_norm(old_weights, new_weights)
    if current_norm <= 1e-12 or target_norm <= 0:
        return [layer.copy() for layer in new_weights]
    scale = min(1.0, float(target_norm) / current_norm)
    return [old + scale * (new - old) for old, new in zip(old_weights, new_weights)]


def local_search_update(
    *,
    model_template: nn.Module,
    old_weights: Weights,
    proxy_buffer,
    device: torch.device,
    fl_lr: float,
    steps: int,
    gamma_scale: float,
    lambda_stealth: float,
    search_batch_size: int,
    state_tail_layers: int,
) -> Weights:
    if steps <= 0 or gamma_scale <= 0:
        return [layer.copy() for layer in old_weights]
    model = build_model_from_template(model_template, old_weights, device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=max(1e-5, float(fl_lr)))
    tail_params = [param for _, param in tail_named_parameters(model, state_tail_layers)]
    origin_tail = [param.detach().clone() for param in tail_params]
    benign_x, benign_y = proxy_buffer.sample(search_batch_size, device)
    benign_loss = F.cross_entropy(model(benign_x), benign_y)
    benign_grads = torch.autograd.grad(benign_loss, tail_params, retain_graph=False, allow_unused=True)
    benign_grad_vec = vectorize_tensors([grad.detach() for grad in benign_grads if grad is not None])
    for _ in range(max(1, int(steps))):
        images, labels = proxy_buffer.sample(search_batch_size, device)
        logits = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(ce_loss):
            break
        tail_params = [param for _, param in tail_named_parameters(model, state_tail_layers)]
        if lambda_stealth > 0:
            update_vec = vectorize_tensors([origin - current for origin, current in zip(origin_tail, tail_params)])
            cosine = F.cosine_similarity(update_vec.unsqueeze(0), benign_grad_vec.unsqueeze(0)).mean()
        else:
            cosine = torch.zeros((), device=device, dtype=ce_loss.dtype)
        objective = (1.0 - lambda_stealth) * ce_loss + lambda_stealth * cosine
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.data.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
    adv_weights = capture_weights(model)
    return [old + float(gamma_scale) * (adv - old) for old, adv in zip(old_weights, adv_weights)]


def train_proxy_model(*, model_template, old_weights, proxy_buffer, device, lr: float, steps: int, batch_size: int) -> Weights:
    params = AttackParameters(gamma_scale=1.0, local_steps=steps, lambda_stealth=0.0, local_search_lr=lr)
    return craft_malicious_update(
        model_template=model_template,
        old_weights=old_weights,
        proxy_buffer=proxy_buffer,
        device=device,
        params=params,
        search_batch_size=batch_size,
        state_tail_layers=1,
    )


def craft_malicious_update(
    *,
    model_template,
    old_weights: Weights,
    proxy_buffer,
    device: torch.device,
    params: AttackParameters,
    search_batch_size: int,
    state_tail_layers: int,
) -> Weights:
    return local_search_update(
        model_template=model_template,
        old_weights=old_weights,
        proxy_buffer=proxy_buffer,
        device=device,
        fl_lr=params.local_search_lr,
        steps=params.local_steps,
        gamma_scale=params.gamma_scale,
        lambda_stealth=params.lambda_stealth,
        search_batch_size=search_batch_size,
        state_tail_layers=state_tail_layers,
    )


@dataclass
class SimulatedRoundResult:
    weights: Weights
    proxy_loss: float
    proxy_acc: float
    malicious_selected: bool
    benign_norm_mean: float
    malicious_norm_mean: float
