"""Deterministic dataset-index partitioning."""

from __future__ import annotations

import numpy as np

from meta_stackelberg.core.random_state import RandomSource


def iid_partition(
    num_samples: int,
    num_clients: int,
    rng: RandomSource,
) -> tuple[tuple[int, ...], ...]:
    """Shuffle indices once and split them into balanced, disjoint clients."""

    if num_clients <= 0:
        raise ValueError('num_clients must be positive')
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')
    if num_samples < num_clients:
        raise ValueError('num_samples must be at least num_clients')
    shuffled = rng.numpy.permutation(num_samples)
    return tuple(
        tuple(int(index) for index in partition)
        for partition in np.array_split(shuffled, num_clients)
    )
