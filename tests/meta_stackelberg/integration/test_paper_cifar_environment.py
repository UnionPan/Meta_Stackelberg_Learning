import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.experiments.paper_cifar_env import PaperCIFAREnvironmentFactory


def _cifar_like(samples=400):
    generator = torch.Generator().manual_seed(7)
    return TensorDataset(
        torch.randn(samples, 3, 32, 32, generator=generator),
        torch.arange(samples) % 10,
    )


def test_cifar_factory_runs_buffer_aware_local_sgd_round() -> None:
    dataset = _cifar_like()
    root = TensorDataset(dataset.tensors[0][:20], dataset.tensors[1][:20])
    factory = PaperCIFAREnvironmentFactory(
        train_dataset=dataset,
        root_dataset=root,
        partition_seed=3,
        model_seed=4,
        workers=20,
        untargeted_attackers=10,
        sample_size=2,
        fl_batch_size=8,
        local_search_batch_size=4,
    )
    env = factory.make(seed=11, horizon=1)
    pending = env.begin_round(np.zeros(3, dtype=np.float32))
    step = env.finish_round(np.array([0.0, -1.0, 0.0], dtype=np.float32))

    assert pending.attacker_observation['model_tail'].size == 5130
    assert step.done
    assert len(step.transition.state_after.global_model.tensors) > len(
        tuple(factory.initial_model.parameters())
    )
    assert np.all(np.isfinite(step.transition.state_after.global_model.vector()))
