import math

import numpy as np
import pytest

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.data import partitioning


def _mean_label_entropy(labels: np.ndarray, partitions: tuple[tuple[int, ...], ...]) -> float:
    entropies = []
    for indices in partitions:
        counts = np.bincount(labels[list(indices)], minlength=4)
        probabilities = counts[counts > 0] / counts.sum()
        entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
    return float(np.mean(entropies))


def test_dirichlet_partition_is_replayable_complete_disjoint_and_label_skewed() -> None:
    labels = np.repeat(np.arange(4), 100)

    first = partitioning.dirichlet_label_partition(
        labels,
        num_clients=8,
        concentration=0.05,
        rng=RandomSource(17),
        min_samples_per_client=10,
    )
    second = partitioning.dirichlet_label_partition(
        labels,
        num_clients=8,
        concentration=0.05,
        rng=RandomSource(17),
        min_samples_per_client=10,
    )

    assert first == second
    assert min(map(len, first)) >= 10
    flattened = tuple(index for client in first for index in client)
    assert len(flattened) == len(labels)
    assert len(set(flattened)) == len(labels)
    assert set(flattened) == set(range(len(labels)))

    iid_entropy = math.log(4.0)
    assert _mean_label_entropy(labels, first) < 0.75 * iid_entropy


@pytest.mark.parametrize(
    ('labels', 'num_clients', 'concentration', 'minimum'),
    [
        ([], 2, 0.5, 1),
        ([0, 1], 0, 0.5, 1),
        ([0, 1], 2, 0.0, 1),
        ([0, 1], 2, 0.5, 0),
        ([0, 1], 2, 0.5, 2),
        ([0.0, float('nan')], 2, 0.5, 1),
    ],
)
def test_dirichlet_partition_rejects_invalid_contract(
    labels: list[int],
    num_clients: int,
    concentration: float,
    minimum: int,
) -> None:
    with pytest.raises(ValueError):
        partitioning.dirichlet_label_partition(
            labels,
            num_clients=num_clients,
            concentration=concentration,
            rng=RandomSource(3),
            min_samples_per_client=minimum,
        )
