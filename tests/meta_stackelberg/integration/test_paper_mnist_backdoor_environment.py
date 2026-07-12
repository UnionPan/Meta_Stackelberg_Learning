from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.experiments.paper_mnist_backdoor_env import (
    PaperMNISTBackdoorEnvironmentFactory,
    make_whitebox_mnist_datasets,
)


def _mnist_like(samples: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples, dtype=torch.long) % 10
    return TensorDataset(inputs, labels)


def test_whitebox_dataset_keeps_all_training_examples_on_clients() -> None:
    train = _mnist_like(400, 7)
    query = _mnist_like(100, 8)

    bundle = make_whitebox_mnist_datasets(
        train_dataset=train,
        held_out_test=query,
        reward_samples=40,
        seed=17,
    )

    assert bundle.client_train is train
    assert len(bundle.client_train) == 400
    assert len(bundle.reward) == 40
    assert len(bundle.query) == 100
    assert set(bundle.reward_indices).issubset(set(range(400)))
    assert bundle.protocol == 'mnist-whitebox-real-data-v1'


def test_whitebox_factory_partitions_every_training_index_once() -> None:
    bundle = make_whitebox_mnist_datasets(
        train_dataset=_mnist_like(400, 7),
        held_out_test=_mnist_like(100, 8),
        reward_samples=40,
        seed=17,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=bundle,
        partition_seed=18,
        model_seed=99,
        workers=20,
        backdoor_attackers=2,
        sample_size=10,
        fl_batch_size=16,
        malicious_batch_size=8,
    )

    assigned = [
        int(index)
        for client_id in range(20)
        for index in factory.client_datasets[client_id].indices
    ]
    assert len(assigned) == 400
    assert len(set(assigned)) == 400
    assert set(assigned) == set(range(400))
    assert factory.malicious_ids == frozenset({0, 1})


def test_whitebox_factory_runs_one_real_mnist_backdoor_round() -> None:
    bundle = make_whitebox_mnist_datasets(
        train_dataset=_mnist_like(400, 7),
        held_out_test=_mnist_like(100, 8),
        reward_samples=40,
        seed=17,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=bundle,
        partition_seed=18,
        model_seed=99,
        workers=20,
        backdoor_attackers=2,
        sample_size=10,
        fl_batch_size=16,
        malicious_batch_size=8,
    )
    env = factory.make(seed=19, horizon=1)

    pending = env.begin_round(np.zeros(3, dtype=np.float32))
    step = env.finish_round(np.array([0.0, 0.0, -1.0], dtype=np.float32))

    assert pending.attacker_observation['model_tail'].size == 1290
    assert step.done
    assert step.transition.state_after.round_index == 1
    assert len(step.transition.sampled_clients) == 10
    assert step.transition.private_diagnostics['source_class'] == 1
    assert step.transition.private_diagnostics['target_class'] == 7


def test_whitebox_factory_reuses_partition_and_initial_model() -> None:
    bundle = make_whitebox_mnist_datasets(
        train_dataset=_mnist_like(400, 7),
        held_out_test=_mnist_like(100, 8),
        reward_samples=40,
        seed=17,
    )
    factory = PaperMNISTBackdoorEnvironmentFactory(
        datasets=bundle,
        partition_seed=18,
        model_seed=99,
        workers=20,
        backdoor_attackers=2,
        sample_size=10,
    )

    first = factory.make(seed=1, horizon=1)
    second = factory.make(seed=2, horizon=1)

    np.testing.assert_array_equal(
        first.state.global_model.vector(), second.state.global_model.vector(),
    )
    assert (
        first.malicious_client_datasets[0].indices
        == second.malicious_client_datasets[0].indices
    )
