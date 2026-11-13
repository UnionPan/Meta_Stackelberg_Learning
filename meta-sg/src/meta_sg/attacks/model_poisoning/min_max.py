"""Design: preserve the existing gradient-space Min-Max search and delta sign.
Data flow: negate honest deltas -> mean and deviation -> distance-constrained
search -> negate the crafted gradient -> colluding model-delta submissions.
This is the repository's multi-local-step adaptation of the original attack.
"""

from collections.abc import Sequence

import torch

from meta_sg.attacks.model_poisoning._common import repeat_update
from meta_sg.federated.state import ModelState, stack_updates, unflatten_update


@torch.no_grad()
def min_max_updates(
    benign_updates: Sequence[ModelState],
    *,
    num_malicious: int,
    direction: str = 'std',
    initial_lambda: float = 10.0,
    tolerance: float = 1e-5,
) -> tuple[list[ModelState], dict]:
    if len(benign_updates) < 2:
        raise ValueError('Min-Max requires at least two honest reference updates')
    gradients = -stack_updates(benign_updates)
    mean = gradients.mean(dim=0)
    if direction == 'std':
        deviation = gradients.std(dim=0, correction=1)
    elif direction == 'sign':
        deviation = mean.sign()
    elif direction == 'unit_vec':
        deviation = mean / mean.norm() if mean.norm() > 0 else torch.zeros_like(mean)
    else:
        raise ValueError(f'unknown Min-Max direction: {direction}')
    limit = torch.stack([torch.norm(gradients - row, dim=1).square().max() for row in gradients]).max()
    degenerate = bool(deviation.norm() == 0 or limit == 0)
    lam = gradients.new_tensor([initial_lambda])
    step, success, iterations = lam, gradients.new_tensor([0.0]), 0
    if not degenerate:
        while bool(torch.abs(success - lam) > tolerance):
            if iterations >= 128:
                raise RuntimeError('Min-Max search did not converge')
            candidate = mean - lam * deviation
            maximum = torch.norm(gradients - candidate, dim=1).square().max()
            if maximum <= limit:
                success = lam
                lam = lam + step / 2
            else:
                lam = lam - step / 2
            step = step / 2
            iterations += 1
    candidate = mean - success * deviation
    maximum = torch.norm(gradients - candidate, dim=1).square().max()
    update = unflatten_update(-candidate, benign_updates[0])
    return repeat_update(update, num_malicious), {
        'lambda': float(success.item()), 'max_benign_distance_sq': float(limit),
        'max_malicious_distance_sq': float(maximum), 'iterations': iterations,
        'direction': direction, 'degenerate': degenerate,
    }
