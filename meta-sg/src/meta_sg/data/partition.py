"""Design: partition CPU training indices without copying or transforming images.
Data flow: labels -> class-balanced trusted root and remaining indices ->
seeded IID or Dirichlet client partitions. Test-set indices stay separate.
"""

from collections.abc import Sequence

import numpy as np
import torch


def split_root_indices(
    targets: Sequence[int] | torch.Tensor,
    samples_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trusted root indices and all remaining training indices."""
    targets = torch.as_tensor(targets)
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for label in torch.unique(targets, sorted=True):
        candidates = torch.where(targets == label)[0]
        if len(candidates) < samples_per_class:
            raise ValueError(f'class {int(label)} has fewer than {samples_per_class} samples')
        order = torch.randperm(len(candidates), generator=generator)
        parts.append(candidates[order[:samples_per_class]])
    root = torch.cat(parts)
    remaining = torch.ones(len(targets), dtype=torch.bool)
    remaining[root] = False
    return root, torch.where(remaining)[0]


def partition_iid_indices(
    indices: torch.Tensor,
    num_clients: int,
    seed: int,
) -> list[torch.Tensor]:
    """Shuffle candidate indices and split them into nearly equal client subsets."""
    if len(indices) < num_clients:
        raise ValueError('each client needs at least one training sample')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)
    return list(torch.tensor_split(indices[order], num_clients))


def partition_dirichlet_indices(
    targets: Sequence[int] | torch.Tensor,
    indices: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[torch.Tensor]:
    """Draw per-class client proportions, then allocate samples with a multinomial.

    Smaller alpha usually produces stronger label skew. Retry the full split
    when a client is empty, following the existing experiment's 100-attempt limit.
    Cache class membership once; each attempt shuffles fresh copies of those pools.
    """
    if len(indices) < num_clients:
        raise ValueError('each client needs at least one training sample')
    targets = torch.as_tensor(targets)
    generator = np.random.default_rng(seed)
    candidate_targets = targets[indices]
    class_pools = [
        indices[candidate_targets == label].numpy()
        for label in torch.unique(candidate_targets, sorted=True)
    ]
    concentration = np.full(num_clients, alpha, dtype=np.float64)
    for _ in range(100):
        client_parts: list[list[int]] = [[] for _ in range(num_clients)]
        for pool in class_pools:
            class_indices = pool.copy()
            generator.shuffle(class_indices)
            proportions = generator.dirichlet(concentration)
            counts = generator.multinomial(len(class_indices), proportions)
            offsets = np.cumsum(counts)[:-1]
            for client_id, part in enumerate(np.split(class_indices, offsets)):
                client_parts[client_id].extend(int(index) for index in part)
        if all(client_parts):
            return [
                torch.tensor(generator.permutation(part), dtype=torch.long)
                for part in client_parts
            ]
    raise RuntimeError('failed to produce non-empty Dirichlet client partitions')
