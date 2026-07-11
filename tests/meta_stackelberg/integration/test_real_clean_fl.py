from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundRequest, RoundState, RoundTransition


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _linearly_separable_dataset() -> TensorDataset:
    offsets = torch.linspace(0.0, 1.0, steps=40)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = torch.stack((1.0 + offsets, 0.5 + 0.25 * offsets), dim=1)
    features = torch.cat((negative, positive), dim=0)
    labels = torch.cat((
        torch.zeros(40, dtype=torch.long),
        torch.ones(40, dtype=torch.long),
    ))
    return TensorDataset(features, labels)


@dataclass(frozen=True)
class RunResult:
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    final_parameters: np.ndarray
    metric_history: tuple[tuple[float, float], ...]
    transitions: tuple[RoundTransition, ...]


def _run_clean_fl() -> RunResult:
    dataset = _linearly_separable_dataset()
    partitions = iid_partition(len(dataset), 4, RandomSource(101))
    client_datasets = {
        client_id: Subset(dataset, indices)
        for client_id, indices in enumerate(partitions)
    }
    codec = TorchParameterCodec()
    source = RandomSource(202)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )
    trainer = TorchLocalTrainer(
        model_factory=_model_factory,
        client_datasets=client_datasets,
        codec=codec,
        learning_rate=0.2,
        local_epochs=2,
        batch_size=8,
    )
    engine = RoundEngine(
        sampler=UniformClientSampler(num_clients=4),
        trainer=trainer,
        aggregator=FedAvg(),
        server_optimizer=ServerSGD(),
    )
    evaluator = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=dataset,
        codec=codec,
        batch_size=16,
    )
    initial = evaluator.evaluate(state.global_model)
    history: list[tuple[float, float]] = []
    transitions: list[RoundTransition] = []
    for _ in range(10):
        transition = engine.run_round(
            RoundRequest(
                task_id='linearly-separable-clean',
                state=state,
                sample_size=4,
                server_lr=1.0,
            ),
            source,
        )
        transitions.append(transition)
        state = transition.state_after
        metrics = evaluator.evaluate(state.global_model)
        history.append((metrics.loss, metrics.accuracy))
    final = evaluator.evaluate(state.global_model)
    return RunResult(
        initial_loss=initial.loss,
        final_loss=final.loss,
        initial_accuracy=initial.accuracy,
        final_accuracy=final.accuracy,
        final_parameters=state.global_model.vector(),
        metric_history=tuple(history),
        transitions=tuple(transitions),
    )


def test_real_clean_fl_converges_and_replays_exactly() -> None:
    first = _run_clean_fl()
    second = _run_clean_fl()

    assert first.initial_accuracy == 0.5
    assert first.final_loss < first.initial_loss
    assert first.final_accuracy >= 0.95
    assert all(len(transition.benign_updates) == 4 for transition in first.transitions)
    assert all(transition.malicious_updates == () for transition in first.transitions)
    np.testing.assert_array_equal(first.final_parameters, second.final_parameters)
    assert first.metric_history == second.metric_history
