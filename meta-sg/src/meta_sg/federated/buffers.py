"""Design: synchronize model buffers independently of parameter-delta defenses.
Data flow: global buffer snapshot -> reset each local model -> client snapshots ->
equal mean of floating buffers and maximum of integer counters -> global buffers.
Synthetic model-poisoning submissions retain the global buffers unchanged.
"""

import torch


def model_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


@torch.no_grad()
def load_model_buffers(model, state):
    for name, value in model.named_buffers():
        value.copy_(state[name])


@torch.no_grad()
def aggregate_buffers(states):
    result = {}
    for name, reference in states[0].items():
        values = torch.stack([state[name] for state in states])
        result[name] = values.mean(dim=0) if reference.is_floating_point() else values.amax(dim=0)
    return result
