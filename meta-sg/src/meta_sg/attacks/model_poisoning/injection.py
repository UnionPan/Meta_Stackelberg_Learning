"""Design: inject crafted model updates using persistent global malicious IDs.
Data flow: sampled identities -> train only honest participants -> craft malicious
updates from their deltas -> restore submissions to the sampled client order.
"""

from meta_sg.attacks._common import active_clients
from meta_sg.federated.client import train_clients


def model_poisoning_round(
    model, base_state, client_loaders, selected_clients, *,
    round_index, steps, learning_rate, device, seed,
    malicious_clients, craft, attack_start=1,
    base_buffers=None, client_buffers=None,
    epochs=None, momentum=0.0, weight_decay=0.0,
):
    malicious = active_clients(selected_clients, malicious_clients, round_index, attack_start)
    honest = [client for client in selected_clients if client not in malicious]
    if malicious and not honest:
        raise ValueError('model-poisoning construction requires selected honest reference updates')
    updates, metrics = train_clients(
        model, base_state, client_loaders, honest, steps=steps,
        learning_rate=learning_rate, device=device, base_buffers=base_buffers, client_buffers=client_buffers,
        epochs=epochs, momentum=momentum, weight_decay=weight_decay,
    )
    diagnostics = {}
    if malicious:
        crafted, diagnostics = craft([updates[client] for client in honest], num_malicious=len(malicious))
        updates.update(zip(malicious, crafted, strict=True))
        if client_buffers is not None:
            client_buffers.update({client: base_buffers for client in malicious})
    return updates, metrics, {'malicious_clients': malicious, **diagnostics}
