"""Design: share fixed-step or full-epoch SGD training across datasets.
Data flow: global parameters and optional buffers -> worker model -> DataLoader batches ->
cross-entropy and SGD -> parameter delta and sample-weighted training loss.
The loader controls sampling; restart it when its current pass is exhausted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from meta_sg.federated.state import ModelState, load_model_state, state_update
from meta_sg.federated.buffers import load_model_buffers, model_buffers


def _training_batches(loader, steps, epochs):
    if epochs is not None:
        for _ in range(epochs):
            yield from loader
        return
    batches = iter(loader)
    for _ in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        yield batch


def train_local_update(
    model: nn.Module,
    base_state: ModelState,
    loader: DataLoader,
    *,
    steps: int | None,
    learning_rate: float,
    device: torch.device | str,
    batch_transform: Callable | None = None,
    base_buffers: dict | None = None,
    epochs: int | None = None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[ModelState, dict[str, float | int | None]]:
    model.to(device)
    load_model_state(model, base_state)
    if base_buffers is not None:
        load_model_buffers(model, base_buffers)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    examples_seen = 0
    actual_steps = 0

    for images, labels in _training_batches(loader, steps, epochs):
        if batch_transform is not None:
            images, labels = batch_transform(images, labels)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        examples_seen += len(labels)
        actual_steps += 1

    update = state_update(base_state, tuple(parameter.detach() for parameter in model.parameters()))
    return update, {
        'loss': total_loss / examples_seen if examples_seen else None,
        'examples_seen': examples_seen,
        'steps': actual_steps,
    }


def train_clients(
    model: nn.Module,
    base_state: ModelState,
    client_loaders: Sequence[DataLoader],
    selected_clients: Sequence[int],
    *,
    steps: int | None,
    learning_rate: float,
    device: torch.device | str,
    base_buffers: dict | None = None,
    client_buffers: dict | None = None,
    epochs: int | None = None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[dict[int, ModelState], list[dict]]:
    """Train selected clients once and retain their global identities."""
    updates, metrics = {}, []
    for client_id in selected_clients:
        update, row = train_local_update(
            model, base_state, client_loaders[client_id], steps=steps,
            learning_rate=learning_rate, device=device, base_buffers=base_buffers,
            epochs=epochs, momentum=momentum, weight_decay=weight_decay,
        )
        if client_buffers is not None:
            client_buffers[client_id] = model_buffers(model)
        updates[client_id] = update
        metrics.append({'client_id': client_id, **row})
    return updates, metrics
