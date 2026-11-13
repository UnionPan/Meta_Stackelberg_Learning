"""Design: share batch poisoning and local SGD between DBA and source-specific BFL.
Data flow: eligible examples -> trigger and relabel a subset -> ordinary client
training -> optional model-replacement scale. Each selected client trains once.
"""

from functools import partial

import torch

from meta_sg.attacks._common import active_clients
from meta_sg.federated.client import train_local_update
from meta_sg.federated.buffers import model_buffers


def poison_batch(
    images, labels, *, trigger, target_label=2, source_label=None,
    poison_fraction=0.5, generator=None,
    poison_selection='random',
):
    eligible = labels.ne(target_label)
    if source_label is not None:
        eligible &= labels.eq(source_label)
    candidates = torch.where(eligible)[0]
    count = min(len(candidates), int(round(len(images) * poison_fraction)))
    if count == 0:
        return images, labels, candidates[:0]
    if poison_selection == 'random':
        selected = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
    elif poison_selection == 'first':
        selected = candidates[:count]
    else:
        raise ValueError('poison_selection must be random or first')
    poisoned_images, poisoned_labels = images.clone(), labels.clone()
    poisoned_images[selected] = trigger(images[selected])
    poisoned_labels[selected] = target_label
    return poisoned_images, poisoned_labels, selected


def train_backdoor_update(
    model, base_state, loader, *, trigger, target_label, source_label,
    poison_fraction, model_replacement_scale, steps, learning_rate, device, seed,
    base_buffers=None,
    epochs=None, momentum=0.0, weight_decay=0.0, poison_selection='random',
):
    generator = torch.Generator().manual_seed(seed)
    poisoned_examples = 0

    def transform(images, labels):
        nonlocal poisoned_examples
        images, labels, selected = poison_batch(
            images, labels, trigger=trigger, target_label=target_label,
            source_label=source_label, poison_fraction=poison_fraction, generator=generator,
            poison_selection=poison_selection,
        )
        poisoned_examples += len(selected)
        return images, labels

    update, metrics = train_local_update(
        model, base_state, loader, steps=steps, learning_rate=learning_rate,
        device=device, batch_transform=transform, base_buffers=base_buffers,
        epochs=epochs, momentum=momentum, weight_decay=weight_decay,
    )
    return tuple(part * model_replacement_scale for part in update), {
        **metrics, 'poisoned_examples': poisoned_examples,
    }


def backdoor_round(
    model, base_state, client_loaders, selected_clients, *,
    round_index, steps, learning_rate, device, seed, malicious_clients, trigger,
    distributed, target_label=2, source_label=None, poison_fraction=0.5,
    model_replacement_scale=10.0, attack_start=1, cyclic=False, malicious_steps=None,
    base_buffers=None, client_buffers=None,
    epochs=None, momentum=0.0, weight_decay=0.0,
    malicious_learning_rate=None, scale_by_malicious_count=False, poison_selection='random',
):
    population = sorted(malicious_clients)
    malicious = active_clients(selected_clients, population, round_index, attack_start)
    if distributed and cyclic and round_index >= attack_start:
        scheduled = population[(round_index - attack_start) % len(population)]
        malicious = [client for client in malicious if client == scheduled]
    scale = model_replacement_scale / len(malicious) if scale_by_malicious_count and malicious else model_replacement_scale
    updates, metrics, assignments = {}, [], {}
    for client_id in selected_clients:
        if client_id in malicious:
            patch = population.index(client_id) % 4 if distributed else None
            update, row = train_backdoor_update(
                model, base_state, client_loaders[client_id],
                trigger=partial(trigger, trigger_index=patch),
                target_label=target_label, source_label=source_label,
                poison_fraction=poison_fraction, model_replacement_scale=scale,
                steps=steps if malicious_steps is None else malicious_steps,
                learning_rate=learning_rate if malicious_learning_rate is None else malicious_learning_rate, device=device,
                seed=seed + round_index * len(client_loaders) + client_id,
                base_buffers=base_buffers,
                epochs=epochs if malicious_steps is None else None,
                momentum=momentum, weight_decay=weight_decay, poison_selection=poison_selection,
            )
            assignments[client_id] = patch
        else:
            update, row = train_local_update(
                model, base_state, client_loaders[client_id], steps=steps,
                learning_rate=learning_rate, device=device, base_buffers=base_buffers,
                epochs=epochs, momentum=momentum, weight_decay=weight_decay,
            )
        if client_buffers is not None:
            client_buffers[client_id] = model_buffers(model)
        updates[client_id] = update
        metrics.append({'client_id': client_id, **row})
    return updates, metrics, {
        'malicious_clients': malicious, 'trigger_assignment': assignments,
        'poison_fraction': poison_fraction, 'model_replacement_scale': scale,
        'scale_by_malicious_count': scale_by_malicious_count, 'poison_selection': poison_selection,
        'malicious_learning_rate': learning_rate if malicious_learning_rate is None else malicious_learning_rate,
    }
