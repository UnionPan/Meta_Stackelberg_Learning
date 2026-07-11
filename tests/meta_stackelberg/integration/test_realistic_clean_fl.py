from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.checkpointing.round_state import load_round_state, save_round_state
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import dirichlet_label_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.models.registry import ModelRegistry
from meta_stackelberg.federated.models.tiny_cnn import TinyImageCNN
from meta_stackelberg.federated.types import RoundRequest, RoundState


def _dataset(seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images = 0.04 * torch.randn(120, 1, 8, 8, generator=generator)
    labels = torch.arange(120, dtype=torch.long) % 2
    images[labels == 0, :, :, 1:3] += 1.0
    images[labels == 1, :, :, 5:7] += 1.0
    return TensorDataset(images, labels)


def _model_factory() -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(43)
        return TinyImageCNN(num_classes=2)


def _partitions(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return dirichlet_label_partition(
        labels,
        num_clients=6,
        concentration=0.3,
        rng=RandomSource(47),
        min_samples_per_client=8,
    )


@dataclass(frozen=True)
class RunResult:
    initial_loss: float
    final_loss: float
    final_accuracy: float
    parameters: np.ndarray
    metrics: tuple[tuple[float, float], ...]
    sampled_clients: tuple[tuple[int, ...], ...]


def _components() -> tuple[RoundEngine, ClassificationEvaluator, RoundState, RandomSource]:
    train_dataset = _dataset(41)
    held_out_dataset = _dataset(42)
    labels = train_dataset.tensors[1].numpy()
    partitions = _partitions(labels)
    client_datasets = {
        client_id: Subset(train_dataset, indices)
        for client_id, indices in enumerate(partitions)
    }
    registry = ModelRegistry()
    registry.register('tiny-image-cnn', _model_factory)
    codec = TorchParameterCodec()
    source = RandomSource(53)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(registry.create('tiny-image-cnn')),
        random_snapshot=source.capture(),
    )
    trainer = TorchLocalTrainer(
        model_factory=lambda: registry.create('tiny-image-cnn'),
        client_datasets=client_datasets,
        codec=codec,
        learning_rate=0.15,
        local_epochs=1,
        batch_size=8,
    )
    engine = RoundEngine(
        sampler=UniformClientSampler(num_clients=6),
        trainer=trainer,
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )
    evaluator = ClassificationEvaluator(
        model_factory=lambda: registry.create('tiny-image-cnn'),
        dataset=held_out_dataset,
        codec=codec,
        batch_size=24,
    )
    return engine, evaluator, state, source


def _run(*, checkpoint_after: int | None, checkpoint_path) -> RunResult:
    engine, evaluator, state, source = _components()
    initial = evaluator.evaluate(state.global_model)
    metrics = []
    sampled_clients = []
    for step in range(10):
        transition = engine.run_round(
            RoundRequest(
                task_id='non-iid-partial-clean-cnn',
                state=state,
                sample_size=3,
            ),
            source,
        )
        assert transition.malicious_updates == ()
        assert len(transition.benign_updates) == 3
        state = transition.state_after
        sampled_clients.append(transition.sampled_clients)
        result = evaluator.evaluate(state.global_model)
        metrics.append((result.loss, result.accuracy))
        if checkpoint_after == step + 1:
            save_round_state(checkpoint_path, state)
            state = load_round_state(checkpoint_path)
            engine, evaluator, _, _ = _components()
            source = RandomSource(999)
    final = evaluator.evaluate(state.global_model)
    return RunResult(
        initial_loss=initial.loss,
        final_loss=final.loss,
        final_accuracy=final.accuracy,
        parameters=state.global_model.vector(),
        metrics=tuple(metrics),
        sampled_clients=tuple(sampled_clients),
    )


def test_non_iid_partial_participation_cnn_learns_and_resumes_exactly(tmp_path) -> None:
    uninterrupted = _run(checkpoint_after=None, checkpoint_path=tmp_path / 'unused.msr')
    resumed = _run(checkpoint_after=4, checkpoint_path=tmp_path / 'round-4.msr')

    assert uninterrupted.final_loss < 0.5 * uninterrupted.initial_loss
    assert uninterrupted.final_accuracy >= 0.95
    assert len(set(uninterrupted.sampled_clients)) > 1
    assert uninterrupted.sampled_clients == resumed.sampled_clients
    assert uninterrupted.metrics == resumed.metrics
    np.testing.assert_array_equal(uninterrupted.parameters, resumed.parameters)


def test_realistic_clean_fl_fixture_is_actually_label_skewed() -> None:
    labels = _dataset(41).tensors[1].numpy()
    dominant_class_fractions = []
    for indices in _partitions(labels):
        counts = np.bincount(labels[list(indices)], minlength=2)
        dominant_class_fractions.append(float(counts.max() / counts.sum()))

    assert float(np.mean(dominant_class_fractions)) >= 0.75
