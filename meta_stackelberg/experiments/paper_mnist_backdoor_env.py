"""Canonical real-data MNIST white-box backdoor environment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.paper_backdoor_bsmg import PaperBackdoorBSMGEnv
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.models.paper_mnist import PaperMNISTCNN
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger
from meta_stackelberg.security.population import FixedMaliciousPopulation


@dataclass(frozen=True)
class WhiteBoxMNISTDatasets:
    """Real client train data, authorized reward view, and isolated query data."""

    client_train: Dataset
    reward: Dataset
    query: Dataset
    reward_indices: tuple[int, ...]
    protocol: str = 'mnist-whitebox-real-data-v1'

    def __post_init__(self) -> None:
        if len(self.client_train) <= 0 or len(self.reward) <= 0 or len(self.query) <= 0:
            raise ValueError('white-box MNIST datasets must not be empty')
        if self.protocol != 'mnist-whitebox-real-data-v1':
            raise ValueError('unsupported white-box MNIST protocol')
        indices = tuple(self.reward_indices)
        if len(indices) != len(self.reward) or len(indices) != len(set(indices)):
            raise ValueError('reward_indices must uniquely identify the reward view')
        if any(index < 0 or index >= len(self.client_train) for index in indices):
            raise ValueError('reward index is outside client training data')
        object.__setattr__(self, 'reward_indices', indices)


def load_whitebox_mnist_datasets(
    root: str | Path,
    *,
    seed: int,
    reward_samples: int = 200,
    download: bool = False,
) -> WhiteBoxMNISTDatasets:
    """Load all 60,000 training examples for clients and an authorized view."""
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import Compose, Normalize, ToTensor
    except ImportError as error:
        raise RuntimeError('torchvision is required for white-box MNIST data') from error
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    location = str(Path(root).expanduser())
    train = MNIST(location, train=True, transform=transform, download=download)
    query = MNIST(location, train=False, transform=transform, download=download)
    return make_whitebox_mnist_datasets(
        train_dataset=train,
        held_out_test=query,
        reward_samples=reward_samples,
        seed=seed,
    )


def make_whitebox_mnist_datasets(
    *,
    train_dataset: Dataset,
    held_out_test: Dataset,
    reward_samples: int,
    seed: int,
) -> WhiteBoxMNISTDatasets:
    if reward_samples <= 0 or reward_samples > len(train_dataset):
        raise ValueError('reward_samples must be within client training size')
    labels = _dataset_labels(train_dataset)
    classes = np.unique(labels)
    if len(classes) != 10:
        raise ValueError('white-box MNIST requires exactly ten classes')
    source = RandomSource(seed)
    selected: list[int] = []
    base, remainder = divmod(reward_samples, len(classes))
    for position, label in enumerate(classes):
        count = base + (1 if position < remainder else 0)
        candidates = np.flatnonzero(labels == label)
        if count > len(candidates):
            raise ValueError('reward_samples exceed available examples for a class')
        if count:
            chosen = source.numpy.choice(candidates, size=count, replace=False)
            selected.extend(int(index) for index in chosen)
    reward_indices = tuple(int(index) for index in source.numpy.permutation(selected))
    return WhiteBoxMNISTDatasets(
        client_train=train_dataset,
        reward=Subset(train_dataset, reward_indices),
        query=held_out_test,
        reward_indices=reward_indices,
    )


class PaperMNISTBackdoorEnvironmentFactory:
    """Reuse one IID 60,000-example partition and initial model across tasks."""

    def __init__(
        self,
        *,
        datasets: WhiteBoxMNISTDatasets,
        partition_seed: int,
        model_seed: int,
        workers: int = 100,
        backdoor_attackers: int = 5,
        sample_size: int = 10,
        fl_batch_size: int = 128,
        local_iterations: int = 1,
        client_learning_rate: float = 0.05,
        malicious_batch_size: int = 128,
        defender_lambda: float = 0.5,
        attacker_lambda: float = 0.5,
    ) -> None:
        if not isinstance(datasets, WhiteBoxMNISTDatasets):
            raise TypeError('datasets must be WhiteBoxMNISTDatasets')
        if workers <= 0 or workers > len(datasets.client_train):
            raise ValueError('workers must be within client training size')
        if backdoor_attackers <= 0 or backdoor_attackers >= workers:
            raise ValueError('backdoor_attackers must be within worker population')
        if sample_size <= 0 or sample_size > workers:
            raise ValueError('sample_size must be within worker population')
        labels = _dataset_labels(datasets.client_train)
        if len(np.unique(labels)) != 10:
            raise ValueError('paper MNIST environment requires exactly ten classes')
        partitions = iid_partition(
            len(datasets.client_train), workers, RandomSource(partition_seed),
        )
        self.datasets = datasets
        self.model_seed = int(model_seed)
        self.workers = int(workers)
        self.backdoor_attackers = int(backdoor_attackers)
        self.sample_size = int(sample_size)
        self.fl_batch_size = int(fl_batch_size)
        self.local_iterations = int(local_iterations)
        self.client_learning_rate = float(client_learning_rate)
        self.malicious_batch_size = int(malicious_batch_size)
        self.defender_lambda = float(defender_lambda)
        self.attacker_lambda = float(attacker_lambda)
        self.client_datasets = {
            client_id: Subset(datasets.client_train, indices)
            for client_id, indices in enumerate(partitions)
        }
        self.malicious_ids = frozenset(range(self.backdoor_attackers))
        self.codec = TorchParameterCodec()
        self.initial_model = _model_factory(self.model_seed)
        self.initial_global_model = self.codec.capture(self.initial_model)
        self.observation_encoder = ModelTailObservationEncoder.from_model(self.initial_model)
        self.benign_trainer = TorchLocalTrainer(
            model_factory=self.model_factory,
            client_datasets=self.client_datasets,
            codec=self.codec,
            learning_rate=self.client_learning_rate,
            local_epochs=self.local_iterations,
            batch_size=self.fl_batch_size,
        )

    def model_factory(self) -> PaperMNISTCNN:
        return _model_factory(self.model_seed)

    @property
    def client_partition_sha256(self) -> str:
        partitions = tuple(
            tuple(int(index) for index in self.client_datasets[client_id].indices)
            for client_id in range(self.workers)
        )
        payload = json.dumps(partitions, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    @property
    def defender_observation_dim(self) -> int:
        tail = sum(
            self.initial_global_model.tensors[index].size
            for index in self.observation_encoder.parameter_indices
        )
        return tail + 1

    @property
    def attacker_observation_dim(self) -> int:
        return self.defender_observation_dim + 1 + 3

    def make(
        self,
        *,
        seed: int,
        horizon: int,
        task_id: str = 'mnist-whitebox-real-data-v1',
    ) -> PaperBackdoorBSMGEnv:
        source = RandomSource(seed)
        state = RoundState(0, self.initial_global_model, source.capture())
        fixture = mnist_global_trigger()
        return PaperBackdoorBSMGEnv(
            task_id=task_id,
            initial_state=state,
            rng=source,
            horizon=horizon,
            sample_size=self.sample_size,
            initial_observed_max_norm=1.0,
            sampler=UniformClientSampler(self.workers),
            benign_trainer=self.benign_trainer,
            population=FixedMaliciousPopulation(self.malicious_ids),
            model_factory=self.model_factory,
            codec=self.codec,
            malicious_client_datasets={
                client_id: self.client_datasets[client_id]
                for client_id in self.malicious_ids
            },
            reward_dataset=self.datasets.reward,
            observation_encoder=self.observation_encoder,
            server_optimizer=ServerSGD(),
            trigger=fixture.trigger,
            source_class=fixture.source_class,
            target_class=fixture.target_class,
            malicious_batch_size=self.malicious_batch_size,
            defender_lambda=self.defender_lambda,
            attacker_lambda=self.attacker_lambda,
        )


def _model_factory(seed: int) -> PaperMNISTCNN:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return PaperMNISTCNN()


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
