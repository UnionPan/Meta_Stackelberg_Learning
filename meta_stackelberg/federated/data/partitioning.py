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


def dirichlet_label_partition(
    labels: np.ndarray | list[int],
    *,
    num_clients: int,
    concentration: float,
    rng: RandomSource,
    min_samples_per_client: int = 1,
    max_attempts: int = 1_000,
) -> tuple[tuple[int, ...], ...]:
    """Partition indices by sampling one client mixture per observed label.

    Smaller ``concentration`` values create more label-skewed clients. Sampling
    is retried when a client would violate the requested minimum size.
    """

    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('labels must be a non-empty one-dimensional sequence')
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('labels must contain integer class ids')
    if num_clients <= 0:
        raise ValueError('num_clients must be positive')
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError('concentration must be finite and positive')
    if min_samples_per_client <= 0:
        raise ValueError('min_samples_per_client must be positive')
    if values.size < num_clients * min_samples_per_client:
        raise ValueError('not enough samples to satisfy the per-client minimum')
    if max_attempts <= 0:
        raise ValueError('max_attempts must be positive')

    class_indices = tuple(np.flatnonzero(values == label) for label in np.unique(values))
    alpha = np.full(num_clients, float(concentration), dtype=np.float64)
    for _ in range(max_attempts):
        clients: list[list[int]] = [[] for _ in range(num_clients)]
        for indices in class_indices:
            shuffled = rng.numpy.permutation(indices)
            proportions = rng.numpy.dirichlet(alpha)
            counts = rng.numpy.multinomial(len(shuffled), proportions)
            offset = 0
            for client_id, count in enumerate(counts):
                end = offset + int(count)
                clients[client_id].extend(int(index) for index in shuffled[offset:end])
                offset = end
        if min(map(len, clients)) >= min_samples_per_client:
            return tuple(
                tuple(int(index) for index in rng.numpy.permutation(indices))
                for indices in clients
            )
    raise RuntimeError(
        f'could not satisfy min_samples_per_client={min_samples_per_client} '
        f'within {max_attempts} attempts'
    )


def paper_q_label_partition(
    labels: np.ndarray | list[int],
    *,
    num_clients: int,
    q: float,
    rng: RandomSource,
) -> tuple[tuple[int, ...], ...]:
    """Implement the Meta-SG Appendix C class-group assignment."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('labels must be a non-empty one-dimensional sequence')
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('labels must contain integer class ids')
    classes = np.unique(values)
    class_count = len(classes)
    if class_count < 2:
        raise ValueError('paper q partition requires at least two classes')
    if num_clients <= 0 or num_clients % class_count != 0:
        raise ValueError('num_clients must be positive and divisible by class count')
    if not np.isfinite(q) or q < 1.0 / class_count or q > 1.0:
        raise ValueError('q must be within [1/C, 1]')
    if values.size < num_clients:
        raise ValueError('not enough samples for non-empty clients')
    group_by_label = {int(label): group for group, label in enumerate(classes)}
    groups: list[list[int]] = [[] for _ in range(class_count)]
    off_probability = (1.0 - float(q)) / (class_count - 1)
    for index, label in enumerate(values):
        preferred = group_by_label[int(label)]
        probabilities = np.full(class_count, off_probability, dtype=np.float64)
        probabilities[preferred] = float(q)
        group = int(rng.numpy.choice(class_count, p=probabilities))
        groups[group].append(index)
    clients_per_group = num_clients // class_count
    if any(len(group) < clients_per_group for group in groups):
        raise RuntimeError('paper q assignment produced an empty client partition')
    result = []
    for group in groups:
        shuffled = rng.numpy.permutation(group)
        result.extend(
            tuple(int(index) for index in client)
            for client in np.array_split(shuffled, clients_per_group)
        )
    return tuple(result)
