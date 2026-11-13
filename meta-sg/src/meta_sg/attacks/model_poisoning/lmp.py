"""Design: search opposite the honest mean's sign using Krum as a selection oracle.
Data flow: honest deltas -> sign direction and RMS scale -> bracket/binary search ->
identical malicious submissions. oracle_bound is independent of submitted count.
"""

from collections.abc import Sequence

import torch

from meta_sg.attacks.model_poisoning._common import repeat_update
from meta_sg.defenses.krum import krum_scores
from meta_sg.federated.state import ModelState, stack_updates, unflatten_update


@torch.no_grad()
def lmp_updates(
    benign_updates: Sequence[ModelState],
    *,
    num_malicious: int,
    oracle_bound: int | None = None,
    search_steps: int = 20,
    maximum_multiplier: float = 100.0,
) -> tuple[list[ModelState], dict]:
    vectors = stack_updates(benign_updates)
    mean = vectors.mean(dim=0)
    direction = mean.sign()
    scale = max(float(mean.norm() / mean.numel() ** 0.5), 1e-12)
    bound = num_malicious if oracle_bound is None else oracle_bound

    def selected(multiplier):
        candidate = mean - scale * multiplier * direction
        submitted = torch.cat([vectors, candidate.repeat(num_malicious, 1)])
        index = int(krum_scores(submitted, bound).argmin())
        return index >= len(vectors), candidate

    high = min(1.0, maximum_multiplier)
    accepted, _ = selected(high)
    if accepted:
        low = high
        while accepted and high < maximum_multiplier:
            low = high
            high = min(2 * high, maximum_multiplier)
            accepted, _ = selected(high)
    else:
        while not accepted and high > 1e-8:
            high /= 2
            accepted, _ = selected(high)
        if not accepted:
            raise RuntimeError('could not find an LMP candidate selected by Krum')
        low = high
        high = min(2 * high, maximum_multiplier)
    for _ in range(search_steps):
        middle = (low + high) / 2
        accepted, _ = selected(middle)
        if accepted:
            low = middle
        else:
            high = middle
    accepted, candidate = selected(low)
    update = unflatten_update(candidate, benign_updates[0])
    return repeat_update(update, num_malicious), {
        'multiplier': low, 'oracle_bound': bound, 'selected_lmp': accepted,
    }
