"""Canonical MNIST PaperBSMG environment with Appendix C data semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv
from meta_stackelberg.experiments.data_provenance import (
    PaperDatasetProvenance,
    provided_provenance,
)
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import paper_q_label_partition
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.models.paper_mnist import PaperMNISTCNN
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.population import FixedMaliciousPopulation


@dataclass(frozen=True)
class PaperMNISTDatasets:
    client_train: Dataset
    root: Dataset
    test: Dataset
    root_indices: tuple[int, ...]
    provenance: PaperDatasetProvenance | None = None

    def __post_init__(self) -> None:
        provenance = self.provenance or provided_provenance(
            'MNIST', len(self.client_train),
        )
        if provenance.dataset != 'MNIST':
            raise ValueError('MNIST datasets require MNIST provenance')
        object.__setattr__(self, 'provenance', provenance)


def load_paper_mnist_datasets(
    root: str | Path,
    *,
    seed: int,
    root_samples: int = 100,
    download: bool = False,
) -> PaperMNISTDatasets:
    """Load MNIST and remove the server root subset from client training."""
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError('torchvision is required for paper MNIST data') from error
    location = str(Path(root).expanduser())
    full_train = MNIST(location, train=True, transform=ToTensor(), download=download)
    test = MNIST(location, train=False, transform=ToTensor(), download=download)
    client_train, root_dataset, root_indices = split_paper_root_dataset(
        full_train, root_samples=root_samples, seed=seed,
    )
    return PaperMNISTDatasets(
        client_train,
        root_dataset,
        test,
        root_indices,
        PaperDatasetProvenance(
            'MNIST', 'torchvision', 'none', len(client_train), 0, 0, 0,
        ),
    )


def make_generated_mnist_datasets(
    *,
    simulated_train: Dataset,
    root_dataset: Dataset,
    held_out_test: Dataset,
    strict_paper_size: bool = True,
) -> PaperMNISTDatasets:
    if strict_paper_size and len(simulated_train) != 60_000:
        raise ValueError('paper generated MNIST train set must contain 60,000 samples')
    return PaperMNISTDatasets(
        simulated_train,
        root_dataset,
        held_out_test,
        (),
        PaperDatasetProvenance(
            'MNIST', 'paper-generated', 'cGAN', len(simulated_train),
            5_000, 100, 200,
        ),
    )


def split_paper_root_dataset(
    dataset: Dataset,
    *,
    root_samples: int,
    seed: int,
) -> tuple[Subset, Subset, tuple[int, ...]]:
    if root_samples <= 0 or root_samples >= len(dataset):
        raise ValueError('root_samples must be within (0, dataset size)')
    permutation = RandomSource(seed).numpy.permutation(len(dataset))
    root_indices = tuple(int(index) for index in permutation[:root_samples])
    client_indices = tuple(int(index) for index in permutation[root_samples:])
    return Subset(dataset, client_indices), Subset(dataset, root_indices), root_indices


def make_paper_mnist_env(
    *,
    seed: int,
    horizon: int,
    train_dataset: Dataset,
    root_dataset: Dataset,
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
    task_id: str = 'paper-mnist-meta-sg',
) -> PaperBSMGEnv:
    factory = PaperMNISTEnvironmentFactory(
        train_dataset=train_dataset,
        root_dataset=root_dataset,
        partition_seed=seed + 1,
        model_seed=99,
        workers=workers,
        untargeted_attackers=untargeted_attackers,
        sample_size=sample_size,
        non_iid_q=non_iid_q,
        fl_batch_size=fl_batch_size,
        local_iterations=local_iterations,
        client_learning_rate=client_learning_rate,
        local_search_learning_rate=local_search_learning_rate,
        local_search_batch_size=local_search_batch_size,
        local_search_trajectories=local_search_trajectories,
    )
    return factory.make(seed=seed, horizon=horizon, task_id=task_id)


class PaperMNISTEnvironmentFactory:
    """Reuse one paper-q split and initial model across all trajectories."""

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
        self.train_dataset = train_dataset
        self.root_dataset = root_dataset
        self.model_seed = model_seed
        self.workers = workers
        self.sample_size = sample_size
        self.local_search_learning_rate = local_search_learning_rate
        self.local_search_batch_size = local_search_batch_size
        self.local_search_trajectories = local_search_trajectories
        labels = _dataset_labels(train_dataset)
        class_count = len(np.unique(labels))
        if class_count != 10:
            raise ValueError('paper MNIST environment requires exactly ten classes')
        if workers % class_count != 0:
            raise ValueError('workers must be divisible by ten class groups')
        if untargeted_attackers % class_count != 0:
            raise ValueError('attackers must be evenly divisible across class groups')
        if sample_size <= 0 or sample_size > workers:
            raise ValueError('sample_size must be within worker population')
        partitions = paper_q_label_partition(
            labels,
            num_clients=workers,
            q=non_iid_q,
            rng=RandomSource(partition_seed),
        )
        self.client_datasets = {
            client_id: Subset(train_dataset, indices)
            for client_id, indices in enumerate(partitions)
        }
        clients_per_group = workers // class_count
        attackers_per_group = untargeted_attackers // class_count
        self.malicious_ids = frozenset(
            group * clients_per_group + offset
            for group in range(class_count)
            for offset in range(attackers_per_group)
        )
        malicious_datasets = [
            self.client_datasets[client_id]
            for client_id in sorted(self.malicious_ids)
        ]
        if not malicious_datasets:
            raise ValueError('paper MNIST environment requires malicious clients')
        self.attacker_dataset = ConcatDataset(malicious_datasets)
        self.attacker_num_examples = {
            client_id: len(self.client_datasets[client_id])
            for client_id in self.malicious_ids
        }
        self.codec = TorchParameterCodec()
        self.initial_model = _model_factory(model_seed)
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
        return _model_factory(self.model_seed)

    def make(
        self,
        *,
        seed: int,
        horizon: int,
        task_id: str = 'paper-mnist-meta-sg',
    ) -> PaperBSMGEnv:
        source = RandomSource(seed)
        state = RoundState(0, self.initial_global_model, source.capture())
        return PaperBSMGEnv(
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
            attacker_dataset=self.attacker_dataset,
            attacker_num_examples=self.attacker_num_examples,
            root_dataset=self.root_dataset,
            observation_encoder=self.observation_encoder,
            server_optimizer=ServerSGD(),
            local_search_learning_rate=self.local_search_learning_rate,
            local_search_batch_size=self.local_search_batch_size,
            local_search_trajectories=self.local_search_trajectories,
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
