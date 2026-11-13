"""Design: submit the scaled negative mean of the selected honest updates.
Data flow: honest parameter deltas -> mean -> negative scaling -> colluding copies.
This is an omniscient model-poisoning attack and uses honest updates explicitly.
"""

from collections.abc import Sequence

import torch

from meta_sg.attacks.model_poisoning._common import repeat_update
from meta_sg.federated.state import ModelState


@torch.no_grad()
def ipm_updates(
    benign_updates: Sequence[ModelState],
    *,
    num_malicious: int,
    scaling_factor: float = 5.0,
) -> tuple[list[ModelState], dict]:
    mean = tuple(torch.stack(parts).mean(dim=0) for parts in zip(*benign_updates, strict=True))
    update = tuple(-scaling_factor * part for part in mean)
    return repeat_update(update, num_malicious), {'scaling_factor': scaling_factor}
