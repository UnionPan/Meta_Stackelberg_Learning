"""Canonical CIFAR-10 PaperBSMG environment with ResNet-18 state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv
from meta_stackelberg.experiments.paper_mnist_env import split_paper_root_dataset
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import paper_q_label_partition
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.models.paper_cifar import PaperCIFARResNet18
from meta_stackelberg.federated.models.parameters import TorchModelStateCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.population import FixedMaliciousPopulation


@dataclass(frozen=True)
class PaperCIFARDatasets:
    client_train: Dataset
    root: Dataset
    test: Dataset
    root_indices: tuple[int, ...]


def load_paper_cifar_datasets(
    root: str | Path,
    *,
    seed: int,
    root_samples: int = 200,
    download: bool = False,
) -> PaperCIFARDatasets:
    try:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError('torchvision is required for paper CIFAR-10 data') from error
    location = str(Path(root).expanduser())
    full_train = CIFAR10(location, train=True, transform=ToTensor(), download=download)
    test = CIFAR10(location, train=False, transform=ToTensor(), download=download)
    client_train, root_dataset, root_indices = split_paper_root_dataset(
        full_train, root_samples=root_samples, seed=seed,
    )
    return PaperCIFARDatasets(client_train, root_dataset, test, root_indices)


class PaperCIFAREnvironmentFactory:
    def __init__(
        self,
        *,
        train_dataset: Dataset,
        root_dataset: Dataset,
        partition_seed: int,
        model_seed: int,
        workers: int = 100,
        untargeted_attackers: int = 20,
        sample_size: int = 10,
        non_iid_q: float = 0.5,
        fl_batch_size: int = 128,
        local_iterations: int = 1,
        client_learning_rate: float = 0.05,
        local_search_learning_rate: float = 0.01,
        local_search_batch_size: int = 128,
        local_search_trajectories: int = 1,
    ) -> None:
        labels = _dataset_labels(train_dataset)
        if len(np.unique(labels)) != 10:
            raise ValueError('paper CIFAR environment requires exactly ten classes')
        if workers <= 0 or workers % 10 != 0:
            raise ValueError('workers must be positive and divisible by ten')
        if untargeted_attackers <= 0 or untargeted_attackers % 10 != 0:
            raise ValueError('attackers must be positive and divisible by ten')
        if untargeted_attackers >= workers:
            raise ValueError('attackers must be fewer than workers')
        if sample_size <= 0 or sample_size > workers:
            raise ValueError('sample_size must be within worker population')
        partitions = paper_q_label_partition(
            labels, num_clients=workers, q=non_iid_q,
            rng=RandomSource(partition_seed),
        )
        self.client_datasets = {
            client_id: Subset(train_dataset, indices)
            for client_id, indices in enumerate(partitions)
        }
        clients_per_group = workers // 10
        attackers_per_group = untargeted_attackers // 10
        self.malicious_ids = frozenset(
            group * clients_per_group + offset
            for group in range(10)
            for offset in range(attackers_per_group)
        )
        self.attacker_dataset = ConcatDataset([
            self.client_datasets[client_id]
            for client_id in sorted(self.malicious_ids)
        ])
        self.attacker_num_examples = {
            client_id: len(self.client_datasets[client_id])
            for client_id in self.malicious_ids
        }
        self.root_dataset = root_dataset
        self.workers = workers
        self.sample_size = sample_size
        self.model_seed = model_seed
        self.local_search_learning_rate = local_search_learning_rate
        self.local_search_batch_size = local_search_batch_size
        self.local_search_trajectories = local_search_trajectories
        self.codec = TorchModelStateCodec()
        self.initial_model = self.model_factory()
        self.initial_global_model = self.codec.capture(self.initial_model)
        self.observation_encoder = ModelTailObservationEncoder.from_model(
            self.initial_model,
        )
        self.benign_trainer = TorchLocalTrainer(
            model_factory=self.model_factory,
            client_datasets=self.client_datasets,
            codec=self.codec,
            learning_rate=client_learning_rate,
            local_epochs=local_iterations,
            batch_size=fl_batch_size,
        )

    def model_factory(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.model_seed)
            return PaperCIFARResNet18()

    def make(self, *, seed: int, horizon: int, task_id: str = 'paper-cifar-meta-sg'):
        source = RandomSource(seed)
        return PaperBSMGEnv(
            task_id=task_id,
            initial_state=RoundState(
                0, self.initial_global_model, source.capture(),
            ),
            rng=source,
            horizon=horizon,
            sample_size=self.sample_size,
            initial_observed_max_norm=1.0,
            sampler=UniformClientSampler(self.workers),
            benign_trainer=self.benign_trainer,
            population=FixedMaliciousPopulation(self.malicious_ids),
            model_factory=self.model_factory,
            codec=self.codec,
            attacker_dataset=self.attacker_dataset,
            attacker_num_examples=self.attacker_num_examples,
            root_dataset=self.root_dataset,
            observation_encoder=self.observation_encoder,
            server_optimizer=ServerSGD(),
            local_search_learning_rate=self.local_search_learning_rate,
            local_search_batch_size=self.local_search_batch_size,
            local_search_trajectories=self.local_search_trajectories,
        )


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, 'targets'):
        values = np.asarray(getattr(dataset, 'targets'))
    elif isinstance(dataset, torch.utils.data.TensorDataset) and len(dataset.tensors) >= 2:
        values = dataset.tensors[1].detach().cpu().numpy()
    else:
        values = np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    if values.ndim != 1 or len(values) != len(dataset):
        raise ValueError('dataset labels must be one-dimensional and complete')
    return values.astype(np.int64, copy=False)
