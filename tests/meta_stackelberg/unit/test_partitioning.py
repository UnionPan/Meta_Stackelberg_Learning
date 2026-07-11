import pytest

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.data.partitioning import iid_partition


def test_iid_partition_covers_every_sample_once_with_balanced_sizes() -> None:
    partitions = iid_partition(22, 4, RandomSource(7))

    flattened = [index for partition in partitions for index in partition]
    sizes = [len(partition) for partition in partitions]
    assert sorted(flattened) == list(range(22))
    assert len(flattened) == len(set(flattened))
    assert max(sizes) - min(sizes) <= 1


def test_iid_partition_is_seed_reproducible() -> None:
    first = iid_partition(20, 4, RandomSource(11))
    second = iid_partition(20, 4, RandomSource(11))
    different = iid_partition(20, 4, RandomSource(12))

    assert first == second
    assert first != different


@pytest.mark.parametrize(
    ('num_samples', 'num_clients'),
    [(0, 1), (3, 4), (4, 0)],
)
def test_iid_partition_rejects_invalid_sizes(num_samples: int, num_clients: int) -> None:
    with pytest.raises(ValueError):
        iid_partition(num_samples, num_clients, RandomSource(0))
