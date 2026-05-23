"""Action decoding for the paper-inspired RL backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackdoorAction:
    """Decoded 4-dim backdoor action.

    ``poison_grid_index`` selects one of 11 precomputed poison-rate loaders
    (paper: ``poisoned_train_iter_lis[int(action[0] * 5.5 + 5.5)]``);
    ``poison_frac`` is the matching trigger rate kept for diagnostics.
    """

    poison_grid_index: int
    poison_frac: float
    local_lr: float
    local_epochs: int
    boost: float


def decode_backdoor_action(action) -> BackdoorAction:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size < 4:
        padded = np.zeros(4, dtype=np.float32)
        padded[: values.size] = values
        values = padded
    values = np.clip(values[:4], -1.0, 1.0)
    poison_grid_index = int(np.clip(round(float(values[0]) * 5.5 + 5.5), 0, 10))
    local_lr = float(values[1] * 0.05 + 0.05)
    local_epochs = max(1, min(11, int(values[2] * 5.0 + 6.0)))
    boost = float(values[3] * 5.0 + 5.0)
    return BackdoorAction(
        poison_grid_index=poison_grid_index,
        poison_frac=poison_grid_index / 10.0,
        local_lr=max(0.0, min(0.1, local_lr)),
        local_epochs=local_epochs,
        boost=max(0.0, min(10.0, boost)),
    )
