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


def test_paper_q_partition_is_replayable_complete_and_has_declared_group_bias() -> None:
    labels = np.repeat(np.arange(4), 1000)
    first = partitioning.paper_q_label_partition(
        labels, num_clients=8, q=0.7, rng=RandomSource(21),
    )
    second = partitioning.paper_q_label_partition(
        labels, num_clients=8, q=0.7, rng=RandomSource(21),
    )

    assert first == second
    flattened = [index for client in first for index in client]
    assert sorted(flattened) == list(range(len(labels)))
    for group in range(4):
        group_indices = first[2 * group] + first[2 * group + 1]
        group_labels = labels[list(group_indices)]
        assert np.mean(group_labels == group) == pytest.approx(0.7, abs=0.05)
        assert abs(len(first[2 * group]) - len(first[2 * group + 1])) <= 1


@pytest.mark.parametrize('num_clients,q', [(7, 0.5), (8, 0.24), (8, 1.1)])
def test_paper_q_partition_rejects_invalid_group_contract(num_clients, q) -> None:
    with pytest.raises(ValueError):
        partitioning.paper_q_label_partition(
            np.repeat(np.arange(4), 10),
            num_clients=num_clients,
            q=q,
            rng=RandomSource(1),
        )
