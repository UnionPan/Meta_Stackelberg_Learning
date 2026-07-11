import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.experiments.paper_mnist_env import (
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
