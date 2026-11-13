"""Design: coordinate federated rounds with supplied attack and defense functions.
Data flow: sample clients -> clean or attacked client updates -> aggregate deltas
-> update global parameters -> clean/optional backdoor evaluation -> round record.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from meta_sg.federated.client import train_clients, train_local_update
from meta_sg.federated.buffers import aggregate_buffers, load_model_buffers, model_buffers
from meta_sg.federated.evaluation import evaluate
from meta_sg.federated.state import ModelState, apply_model_update, model_state


def run_federated(
    model: nn.Module,
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader | None,
    aggregate: Callable[..., tuple[ModelState, dict[str, Any]]] | None,
    *,
    rounds: int,
    clients_per_round: int,
    local_steps: int | None,
    learning_rate: float,
    device: torch.device | str,
    seed: int,
    root_loader: DataLoader | None = None,
    root_steps: int = 1,
    attack: Callable | None = None,
    backdoor_evaluation: Callable | None = None,
    local_epochs: int | None = None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    learning_rate_decay: float = 1.0,
    start_round: int | None = None,
    participant_generator: np.random.Generator | None = None,
    defender=None,
) -> Iterator[dict[str, Any]]:
    """Update model in place as the caller consumes round-zero and round records."""
    model.to(device)
    worker = deepcopy(model)
    generator = participant_generator if participant_generator is not None else np.random.default_rng(seed)
    training_arguments = {'epochs': local_epochs, 'momentum': momentum, 'weight_decay': weight_decay}
    if start_round is None:
        yield {
            'round': 0,
            'learning_rate': learning_rate,
            'selected_clients': [],
            'client_metrics': [],
            'root_metrics': None,
            'aggregation': {},
            'attack': {},
            'evaluation': evaluate(model, test_loader, device=device) if test_loader is not None else None,
            'backdoor_evaluation': backdoor_evaluation(model) if backdoor_evaluation is not None else None,
        }
    for round_index in range((start_round or 0) + 1, rounds + 1):
        round_learning_rate = learning_rate * learning_rate_decay ** (round_index - 1)
        selected = generator.choice(len(client_loaders), clients_per_round, replace=False).tolist()
        base_state = model_state(model)
        base_buffers = model_buffers(model)
        client_buffers = {}
        buffer_arguments = {'base_buffers': base_buffers, 'client_buffers': client_buffers} if base_buffers else {}
        attack_metrics = {}
        if attack is None:
            updates, client_metrics = train_clients(
                worker, base_state, client_loaders, selected, steps=local_steps,
                learning_rate=round_learning_rate, device=device, **buffer_arguments, **training_arguments,
            )
        else:
            updates, client_metrics, attack_metrics = attack(
                worker, base_state, client_loaders, selected, round_index=round_index,
                steps=local_steps, learning_rate=round_learning_rate, device=device, seed=seed,
                **buffer_arguments, **training_arguments,
            )
        aggregate_args = {}
        root_metrics = None
        if root_loader is not None:
            root_update, root_metrics = train_local_update(
                worker, base_state, root_loader, steps=root_steps,
                learning_rate=round_learning_rate, device=device, base_buffers=base_buffers,
                momentum=momentum, weight_decay=weight_decay,
            )
            aggregate_args['root_update'] = root_update
        submitted = [updates[client] for client in selected]
        if defender is None:
            update, diagnostics = aggregate(submitted, **aggregate_args)
        else:
            update, diagnostics = defender.aggregate(model, submitted, aggregate_args['root_update'])
        apply_model_update(model, update)
        if base_buffers:
            load_model_buffers(model, aggregate_buffers([client_buffers[client] for client in selected]))
            diagnostics['buffer_aggregation'] = 'mean_float_max_counter'
        if defender is not None:
            diagnostics.update(defender.feedback(model, terminal=round_index == rounds))
        yield {
            'round': round_index,
            'learning_rate': round_learning_rate,
            'selected_clients': selected,
            'client_metrics': client_metrics,
            'root_metrics': root_metrics,
            'aggregation': diagnostics,
            'attack': attack_metrics,
            'evaluation': evaluate(model, test_loader, device=device) if test_loader is not None else None,
            'backdoor_evaluation': backdoor_evaluation(model) if backdoor_evaluation is not None else None,
        }
