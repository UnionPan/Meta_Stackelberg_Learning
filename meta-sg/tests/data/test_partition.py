"""Check balanced root sampling, disjoint client coverage, and reproducible splits."""

import numpy as np
import pytest
import torch

from meta_sg.data import partition_dirichlet_indices, partition_iid_indices, split_root_indices


@pytest.mark.parametrize('as_list', [False, True])
def test_root_split_is_balanced_disjoint_complete_and_reproducible(as_list):
    targets = torch.arange(10).repeat_interleave(20)
    labels = targets.tolist() if as_list else targets
    rng_before = torch.get_rng_state()
    root, remaining = split_root_indices(labels, samples_per_class=3, seed=42)
    again, _ = split_root_indices(labels, samples_per_class=3, seed=42)

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(root, again)
    assert root.numel() == 30
    assert not torch.isin(root, remaining).any()
    assert torch.equal(torch.bincount(targets[root]), torch.full((10,), 3))
    assert torch.equal(torch.cat([root, remaining]).sort().values, torch.arange(len(targets)))


def test_iid_split_preserves_original_indices_and_balances_client_sizes():
    indices = torch.arange(23) * 2 + 100
    rng_before = torch.get_rng_state()
    first = partition_iid_indices(indices, num_clients=5, seed=42)
    second = partition_iid_indices(indices, num_clients=5, seed=42)

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert len(first) == 5
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert max(map(len, first)) - min(map(len, first)) <= 1
    assert torch.equal(torch.cat(first).sort().values, indices.sort().values)


def test_root_split_requires_enough_examples_per_class():
    with pytest.raises(ValueError, match='fewer than 3 samples'):
        split_root_indices([0, 0, 1, 1], samples_per_class=3, seed=42)


def test_iid_split_requires_data_for_each_client():
    with pytest.raises(ValueError, match='at least one training sample'):
        partition_iid_indices(torch.arange(3), num_clients=4, seed=42)


@pytest.mark.parametrize('as_list', [False, True])
def test_dirichlet_split_covers_candidates_and_excludes_root_reproducibly(as_list):
    targets = torch.arange(6).repeat_interleave(100)
    root, remaining = split_root_indices(targets, samples_per_class=3, seed=42)
    labels = targets.tolist() if as_list else targets
    rng_before = np.random.get_state()
    first = partition_dirichlet_indices(labels, remaining, num_clients=8, alpha=0.3, seed=42)
    second = partition_dirichlet_indices(labels, remaining, num_clients=8, alpha=0.3, seed=42)
    rng_after = np.random.get_state()

    assert len(first) == 8
    assert all(part.numel() > 0 and part.dtype == torch.int64 for part in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert torch.equal(torch.cat(first).sort().values, remaining)
    assert not torch.isin(torch.cat(first), root).any()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])


def test_smaller_alpha_produces_stronger_label_skew():
    targets = torch.arange(4).repeat_interleave(1000)
    indices = torch.arange(len(targets))
    skewed = partition_dirichlet_indices(targets, indices, num_clients=8, alpha=0.1, seed=42)
    similar = partition_dirichlet_indices(targets, indices, num_clients=8, alpha=100, seed=42)

    def mean_dominant_share(partitions):
        return sum(
            torch.bincount(targets[part], minlength=4).max().item() / len(part)
            for part in partitions
        ) / len(partitions)

    assert mean_dominant_share(skewed) > mean_dominant_share(similar) + 0.2


def test_dirichlet_split_requires_data_for_each_client():
    with pytest.raises(ValueError, match='at least one training sample'):
        partition_dirichlet_indices([0, 0, 1], torch.arange(3), num_clients=4, alpha=0.5, seed=42)
