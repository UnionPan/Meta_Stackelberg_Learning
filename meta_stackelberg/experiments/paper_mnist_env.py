"""Canonical MNIST PaperBSMG environment with Appendix C data semantics."""

from __future__ import annotations

from dataclasses import dataclass
import copy
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
from meta_stackelberg.federated.clients.sampling import (
    BenignReferenceClientSampler,
    UniformClientSampler,
)
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import (
    iid_partition,
    paper_q_label_partition,
)
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
    materialize: bool = False,
    normalization: str = 'none',
) -> PaperMNISTDatasets:
    """Load MNIST and remove the server root subset from client training."""
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import Compose, Normalize, ToTensor
    except ImportError as error:
        raise RuntimeError('torchvision is required for paper MNIST data') from error
    location = str(Path(root).expanduser())
    if normalization not in {'none', 'standard'}:
        raise ValueError('normalization must be none or standard')
    transforms = [ToTensor()]
    if normalization == 'standard':
        transforms.append(Normalize((0.1307,), (0.3081,)))
    transform = Compose(transforms)
    full_train = MNIST(location, train=True, transform=transform, download=download)
    test = MNIST(location, train=False, transform=transform, download=download)
    if materialize:
        full_train = _materialize_mnist(full_train, normalization=normalization)
        test = _materialize_mnist(test, normalization=normalization)
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
    partition_mode: str = 'paper-q',
    fl_batch_size: int = 128,
    local_iterations: int = 1,
    client_learning_rate: float = 0.05,
    local_search_learning_rate: float = 0.01,
    local_search_batch_size: int = 128,
    local_search_trajectories: int = 1,
    local_search_gradient_norm_cap: float = 1.0,
    device: str | torch.device = 'cpu',
    post_defense_mode: str = 'neuroclip',
    parallel_clients: int = 1,
    defender_alpha_floor_ratio: float = 0.0,
    defender_norm_reference: str = 'max',
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
        partition_mode=partition_mode,
        fl_batch_size=fl_batch_size,
        local_iterations=local_iterations,
        client_learning_rate=client_learning_rate,
        local_search_learning_rate=local_search_learning_rate,
        local_search_batch_size=local_search_batch_size,
        local_search_trajectories=local_search_trajectories,
        local_search_gradient_norm_cap=local_search_gradient_norm_cap,
        device=device,
        post_defense_mode=post_defense_mode,
        parallel_clients=parallel_clients,
        defender_alpha_floor_ratio=defender_alpha_floor_ratio,
        defender_norm_reference=defender_norm_reference,
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
        partition_mode: str = 'paper-q',
        fl_batch_size: int = 128,
        local_iterations: int = 1,
        client_learning_rate: float = 0.05,
        local_search_learning_rate: float = 0.01,
        local_search_batch_size: int = 128,
        local_search_trajectories: int = 1,
        local_search_gradient_norm_cap: float = 1.0,
        device: str | torch.device = 'cpu',
        post_defense_mode: str = 'neuroclip',
        parallel_clients: int = 1,
        defender_alpha_floor_ratio: float = 0.0,
        defender_norm_reference: str = 'max',
    ) -> None:
        self.train_dataset = train_dataset
        self.root_dataset = root_dataset
        self.model_seed = model_seed
        self.workers = workers
        self.sample_size = sample_size
        self.local_search_learning_rate = local_search_learning_rate
        self.local_search_batch_size = local_search_batch_size
        self.local_search_trajectories = local_search_trajectories
        self.local_search_gradient_norm_cap = local_search_gradient_norm_cap
        self.device = torch.device(device)
        if partition_mode not in {'iid', 'paper-q'}:
            raise ValueError('partition_mode must be iid or paper-q')
        self.partition_mode = partition_mode
        self.non_iid_q = float(non_iid_q)
        self.defender_alpha_floor_ratio = float(defender_alpha_floor_ratio)
        if defender_norm_reference not in {'max', 'median'}:
            raise ValueError(
                'defender_norm_reference must be max or median',
            )
        self.defender_norm_reference = defender_norm_reference
        if post_defense_mode not in {'neuroclip', 'identity'}:
            raise ValueError(
                'post_defense_mode must be neuroclip or identity'
            )
        self.post_defense_mode = post_defense_mode
        if (
            isinstance(parallel_clients, bool)
            or not isinstance(parallel_clients, int)
            or parallel_clients <= 0
        ):
            raise ValueError('parallel_clients must be a positive integer')
        self.parallel_clients = min(parallel_clients, sample_size)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {self.device} is not available')
        labels = _dataset_labels(train_dataset)
        class_count = len(np.unique(labels))
        if class_count != 10:
            raise ValueError('paper MNIST environment requires exactly ten classes')
        if workers % class_count != 0:
            raise ValueError('workers must be divisible by ten class groups')
        if sample_size <= 0 or sample_size > workers:
            raise ValueError('sample_size must be within worker population')
        partitions = (
            iid_partition(
                len(labels), workers, RandomSource(partition_seed),
            )
            if partition_mode == 'iid'
            else paper_q_label_partition(
                labels,
                num_clients=workers,
                q=non_iid_q,
                rng=RandomSource(partition_seed),
            )
        )
        self.client_datasets = {
            client_id: Subset(train_dataset, indices)
            for client_id, indices in enumerate(partitions)
        }
        clients_per_group = workers // class_count
        attackers_per_group, attacker_remainder = divmod(
            untargeted_attackers, class_count,
        )
        self.malicious_ids = frozenset(
            group * clients_per_group + offset
            for group in range(class_count)
            for offset in range(
                attackers_per_group + (group < attacker_remainder)
            )
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
            batch_size=fl_batch_size,
            local_steps=local_iterations,
            device=self.device,
            reuse_workspace=True,
        )

    def model_factory(self):
        return copy.deepcopy(self.initial_model).to(self.device)

    def make(
        self,
        *,
        seed: int,
        horizon: int,
        task_id: str = 'paper-mnist-meta-sg',
        malicious_ids=None,
        attack_generator_factory=None,
    ) -> PaperBSMGEnv:
        source = RandomSource(seed)
        state = RoundState(0, self.initial_global_model, source.capture())
        selected_malicious_ids = (
            self.malicious_ids
            if malicious_ids is None
            else frozenset(malicious_ids)
        )
        if not selected_malicious_ids.issubset(self.malicious_ids):
            raise ValueError('evaluation malicious ids must use the declared population')
        sampler = (
            BenignReferenceClientSampler(
                self.workers, selected_malicious_ids,
            )
            if selected_malicious_ids
            else UniformClientSampler(self.workers)
        )
        return PaperBSMGEnv(
            task_id=task_id,
            initial_state=state,
            rng=source,
            horizon=horizon,
            sample_size=self.sample_size,
            initial_observed_max_norm=1.0,
            sampler=sampler,
            benign_trainer=self.benign_trainer,
            population=FixedMaliciousPopulation(selected_malicious_ids),
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
            local_search_gradient_norm_cap=self.local_search_gradient_norm_cap,
            device=self.device,
            post_defense_factory=(
                _identity_post_defense
                if self.post_defense_mode == 'identity'
                else None
            ),
            defender_alpha_floor_ratio=self.defender_alpha_floor_ratio,
            defender_norm_reference=self.defender_norm_reference,
            reuse_post_defense_loss=self.post_defense_mode == 'identity',
            parallel_clients=self.parallel_clients,
            attack_generator_factory=attack_generator_factory,
        )


def _identity_post_defense(model, epsilon):
    del epsilon
    return model


def _materialize_mnist(
    dataset: Dataset,
    *,
    normalization: str = 'none',
) -> torch.utils.data.TensorDataset:
    data = getattr(dataset, 'data', None)
    targets = getattr(dataset, 'targets', None)
    if not isinstance(data, torch.Tensor) or not isinstance(
        targets, torch.Tensor,
    ):
        raise TypeError('materialized MNIST requires tensor data and targets')
    inputs = data.unsqueeze(1).to(dtype=torch.float32).div_(255.0)
    if normalization == 'standard':
        inputs.sub_(0.1307).div_(0.3081)
    elif normalization != 'none':
        raise ValueError('normalization must be none or standard')
    return torch.utils.data.TensorDataset(inputs, targets.to(dtype=torch.long))


def _model_factory(seed: int) -> PaperMNISTCNN:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return PaperMNISTCNN()


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        values = _dataset_labels(dataset.dataset)[np.asarray(dataset.indices)]
    elif hasattr(dataset, 'targets'):
        values = np.asarray(getattr(dataset, 'targets'))
    elif isinstance(dataset, torch.utils.data.TensorDataset) and len(dataset.tensors) >= 2:
        values = dataset.tensors[1].detach().cpu().numpy()
    else:
        values = np.asarray([int(dataset[index][1]) for index in range(len(dataset))])
    if values.ndim != 1 or len(values) != len(dataset):
        raise ValueError('dataset labels must be one-dimensional and complete')
    return values.astype(np.int64, copy=False)
