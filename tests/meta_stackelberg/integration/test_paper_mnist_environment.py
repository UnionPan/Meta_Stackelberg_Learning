import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.federated.clients.sampling import (
    BenignReferenceClientSampler,
)
from meta_stackelberg.environments.paper_bsmg import _defender_norm_reference
from meta_stackelberg.experiments.paper_mnist_env import (
    PaperMNISTEnvironmentFactory,
    make_paper_mnist_env,
    split_paper_root_dataset,
)


def _mnist_like(samples=400):
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples) % 10
    return TensorDataset(inputs, labels)


def test_mnist_factory_runs_real_local_sgd_paper_q_and_rl_attack_round() -> None:
    dataset = _mnist_like()
    env = make_paper_mnist_env(
        seed=19,
        horizon=1,
        train_dataset=dataset,
        root_dataset=TensorDataset(dataset.tensors[0][:40], dataset.tensors[1][:40]),
        workers=20,
        untargeted_attackers=10,
        sample_size=10,
        non_iid_q=0.5,
        fl_batch_size=16,
        local_iterations=1,
        client_learning_rate=0.05,
        local_search_batch_size=8,
    )

    pending = env.begin_round(np.zeros(3, dtype=np.float32))
    assert pending.attacker_observation['model_tail'].size == 1290
    step = env.finish_round(np.array([0.0, -1.0, 0.0], dtype=np.float32))

    assert step.done
    assert step.transition.state_after.round_index == 1
    assert len(step.transition.sampled_clients) == 10
    assert step.transition.private_diagnostics['malicious_client_count'] >= 0


def test_root_split_is_seeded_disjoint_and_removed_from_client_training() -> None:
    dataset = _mnist_like(100)
    client, root, indices = split_paper_root_dataset(
        dataset, root_samples=10, seed=5,
    )
    replay = split_paper_root_dataset(dataset, root_samples=10, seed=5)
    assert len(client) == 90 and len(root) == 10
    assert indices == replay[2]
    assert set(client.indices).isdisjoint(root.indices)
    assert set(client.indices) | set(root.indices) == set(range(100))


def test_factory_reuses_fixed_partition_and_initial_model_across_rollout_seeds() -> None:
    dataset = _mnist_like()
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=dataset,
        root_dataset=TensorDataset(dataset.tensors[0][:40], dataset.tensors[1][:40]),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=10,
        sample_size=10,
        fl_batch_size=16,
        local_search_batch_size=8,
    )
    first = factory.make(seed=1, horizon=1)
    second = factory.make(seed=2, horizon=1)
    np.testing.assert_array_equal(
        first.state.global_model.vector(), second.state.global_model.vector(),
    )
    assert first.benign_trainer.client_datasets[0].indices == second.benign_trainer.client_datasets[0].indices


def test_factory_supports_four_attackers_across_twenty_workers() -> None:
    dataset = _mnist_like()
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=dataset,
        root_dataset=TensorDataset(
            dataset.tensors[0][:40], dataset.tensors[1][:40],
        ),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=4,
        sample_size=4,
        fl_batch_size=16,
        local_iterations=1,
        local_search_batch_size=8,
    )

    assert len(factory.malicious_ids) == 4
    assert factory.benign_trainer.local_steps == 1
    assert factory.benign_trainer.local_epochs is None
    attack_env = factory.make(seed=1, horizon=1)
    assert isinstance(attack_env.sampler, BenignReferenceClientSampler)


def test_factory_supports_iid_partition_and_alpha_guard() -> None:
    dataset = _mnist_like()
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=dataset,
        root_dataset=TensorDataset(
            dataset.tensors[0][:40], dataset.tensors[1][:40],
        ),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=4,
        sample_size=4,
        partition_mode='iid',
        defender_alpha_floor_ratio=0.1,
        fl_batch_size=16,
        local_search_batch_size=8,
    )

    env = factory.make(seed=1, horizon=1, malicious_ids=())
    assert factory.partition_mode == 'iid'
    action = env.defender_codec.decode(
        np.array([-1.0, 0.0, 0.0]), observed_max_norm=2.0,
    )
    assert action.alpha == pytest.approx(0.2)


def test_median_norm_reference_rejects_single_extreme_update_scale() -> None:
    dataset = _mnist_like()
    common = dict(
        train_dataset=dataset,
        root_dataset=TensorDataset(
            dataset.tensors[0][:40], dataset.tensors[1][:40],
        ),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=4,
        sample_size=4,
        partition_mode='iid',
        fl_batch_size=16,
        local_search_batch_size=8,
    )
    maximum = PaperMNISTEnvironmentFactory(
        **common, defender_norm_reference='max',
    ).make(seed=1, horizon=1, malicious_ids=())
    median = PaperMNISTEnvironmentFactory(
        **common, defender_norm_reference='median',
    ).make(seed=1, horizon=1, malicious_ids=())

    assert maximum.defender_norm_reference == 'max'
    assert median.defender_norm_reference == 'median'
    assert _defender_norm_reference(
        (1.0, 1.1, 1.2, 1_000.0), 'max',
    ) == 1_000.0
    assert _defender_norm_reference(
        (1.0, 1.1, 1.2, 1_000.0), 'median',
    ) == pytest.approx(1.15)


def test_factory_identity_mode_keeps_neuroclip_as_default_configuration() -> None:
    dataset = _mnist_like()
    common = dict(
        train_dataset=dataset,
        root_dataset=TensorDataset(
            dataset.tensors[0][:40], dataset.tensors[1][:40],
        ),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=4,
        sample_size=4,
        fl_batch_size=16,
        local_search_batch_size=8,
    )
    paper_default = PaperMNISTEnvironmentFactory(**common)
    global_identity = PaperMNISTEnvironmentFactory(
        **common, post_defense_mode='identity',
    )

    assert paper_default.post_defense_mode == 'neuroclip'
    assert not paper_default.make(seed=1, horizon=1).reuse_post_defense_loss
    assert global_identity.make(
        seed=1, horizon=1,
    ).reuse_post_defense_loss


def test_parallel_client_updates_match_serial_rounds_exactly() -> None:
    dataset = _mnist_like()
    common = dict(
        train_dataset=dataset,
        root_dataset=TensorDataset(
            dataset.tensors[0][:40], dataset.tensors[1][:40],
        ),
        partition_seed=8,
        model_seed=9,
        workers=20,
        untargeted_attackers=4,
        sample_size=4,
        fl_batch_size=16,
        local_iterations=1,
        local_search_batch_size=8,
        post_defense_mode='identity',
    )
    serial = PaperMNISTEnvironmentFactory(
        **common, parallel_clients=1,
    ).make(seed=31, horizon=2, malicious_ids=())
    parallel = PaperMNISTEnvironmentFactory(
        **common, parallel_clients=4,
    ).make(seed=31, horizon=2, malicious_ids=())
    defender_action = np.zeros(3, dtype=np.float32)
    attacker_action = np.zeros(3, dtype=np.float32)

    for _ in range(2):
        serial_pending = serial.begin_round(defender_action)
        parallel_pending = parallel.begin_round(defender_action)
        assert serial_pending.round_index == parallel_pending.round_index
        assert serial_pending.defender_action == parallel_pending.defender_action
        for key in serial_pending.attacker_observation:
            np.testing.assert_array_equal(
                serial_pending.attacker_observation[key],
                parallel_pending.attacker_observation[key],
            )
        assert serial.finish_round(attacker_action) == parallel.finish_round(
            attacker_action,
        )
