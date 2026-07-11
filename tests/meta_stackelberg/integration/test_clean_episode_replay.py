from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.checkpointing.round_state import load_round_state, save_round_state
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec, FederatedTrajectory
from meta_stackelberg.federated.evaluation.classification import (
    ClassificationEvaluator,
    ClassificationMetrics,
)
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundRequest, RoundState, RoundTransition


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _dataset(offset_start: float) -> TensorDataset:
    offsets = torch.linspace(offset_start, offset_start + 1.0, steps=40)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = torch.stack((1.0 + offsets, 0.5 + 0.25 * offsets), dim=1)
    features = torch.cat((negative, positive), dim=0)
    labels = torch.cat((
        torch.zeros(40, dtype=torch.long),
        torch.ones(40, dtype=torch.long),
    ))
    return TensorDataset(features, labels)


@dataclass(frozen=True)
class Components:
    engine: RoundEngine
    evaluator: ClassificationEvaluator
    state: RoundState
    source: RandomSource


def _components() -> Components:
    train_dataset = _dataset(0.0)
    held_out_dataset = _dataset(0.25)
    partitions = iid_partition(len(train_dataset), 4, RandomSource(101))
    client_datasets = {
        client_id: Subset(train_dataset, indices)
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
    return Components(
        engine=RoundEngine(
            sampler=UniformClientSampler(num_clients=4),
            trainer=trainer,
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        ),
        evaluator=ClassificationEvaluator(
            model_factory=_model_factory,
            dataset=held_out_dataset,
            codec=codec,
            batch_size=16,
        ),
        state=state,
        source=source,
    )


def _spec(state: RoundState, horizon: int) -> EpisodeSpec:
    return EpisodeSpec(
        task_id='clean-episode-replay',
        horizon=horizon,
        sample_size=2,
        server_lr=1.0,
        initial_state=state,
    )


def _manual_loop(components: Components, horizon: int) -> tuple[RoundTransition, ...]:
    transitions = []
    state = components.state
    for _ in range(horizon):
        transition = components.engine.run_round(
            RoundRequest(
                task_id='clean-episode-replay',
                state=state,
                sample_size=2,
                server_lr=1.0,
            ),
            components.source,
        )
        transitions.append(transition)
        state = transition.state_after
    return tuple(transitions)


def _metrics(evaluator: ClassificationEvaluator, trajectory: FederatedTrajectory) -> ClassificationMetrics:
    return evaluator.evaluate(trajectory.final_state.global_model)


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    if left.torch_cpu_state is None or right.torch_cpu_state is None:
        assert left.torch_cpu_state is None and right.torch_cpu_state is None
    else:
        assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_states_equal(left: RoundState, right: RoundState) -> None:
    assert left.round_index == right.round_index
    assert dict(left.component_states) == dict(right.component_states)
    assert len(left.global_model.tensors) == len(right.global_model.tensors)
    for left_tensor, right_tensor in zip(
        left.global_model.tensors,
        right.global_model.tensors,
    ):
        np.testing.assert_array_equal(left_tensor, right_tensor)
    _assert_snapshots_equal(left.random_snapshot, right.random_snapshot)


def _assert_updates_equal(left: ClientUpdate, right: ClientUpdate) -> None:
    assert left.client_id == right.client_id
    assert left.num_examples == right.num_examples
    assert left.is_malicious == right.is_malicious
    assert dict(left.metadata) == dict(right.metadata)
    for left_tensor, right_tensor in zip(left.delta.tensors, right.delta.tensors):
        np.testing.assert_array_equal(left_tensor, right_tensor)


def _assert_transitions_equal(left: RoundTransition, right: RoundTransition) -> None:
    assert left.task_id == right.task_id
    assert left.sampled_clients == right.sampled_clients
    assert len(left.benign_updates) == len(right.benign_updates)
    assert len(left.malicious_updates) == len(right.malicious_updates)
    _assert_states_equal(left.state_before, right.state_before)
    _assert_states_equal(left.state_after, right.state_after)
    for left_update, right_update in zip(left.benign_updates, right.benign_updates):
        _assert_updates_equal(left_update, right_update)
    for left_update, right_update in zip(left.malicious_updates, right.malicious_updates):
        _assert_updates_equal(left_update, right_update)
    for left_tensor, right_tensor in zip(
        left.aggregate_delta.tensors,
        right.aggregate_delta.tensors,
    ):
        np.testing.assert_array_equal(left_tensor, right_tensor)
    assert dict(left.public_signals) == dict(right.public_signals)
    assert dict(left.private_diagnostics) == dict(right.private_diagnostics)


def test_real_clean_episode_matches_manual_loop_and_resumes_exactly(tmp_path) -> None:
    full_components = _components()
    initial_metrics = full_components.evaluator.evaluate(full_components.state.global_model)
    full = EpisodeRunner(full_components.engine).run(
        _spec(full_components.state, horizon=8),
        full_components.source,
    )

    prefix_components = _components()
    prefix = EpisodeRunner(prefix_components.engine).run(
        _spec(prefix_components.state, horizon=3),
        prefix_components.source,
    )
    checkpoint = tmp_path / 'round-3.msr'
    save_round_state(checkpoint, prefix.final_state)
    restored = load_round_state(checkpoint)
    suffix_components = _components()
    suffix = EpisodeRunner(suffix_components.engine).run(
        _spec(restored, horizon=5),
        RandomSource(999),
    )

    manual_components = _components()
    manual = _manual_loop(manual_components, horizon=8)
    resumed = prefix.transitions + suffix.transitions

    assert len(full.transitions) == 8
    assert [step.sampled_clients for step in full.transitions] == [
        step.sampled_clients for step in resumed
    ]
    assert [step.sampled_clients for step in full.transitions] == [
        step.sampled_clients for step in manual
    ]
    assert all(step.malicious_updates == () for step in full.transitions)
    for full_step, resumed_step in zip(full.transitions, resumed):
        _assert_transitions_equal(full_step, resumed_step)
    for full_step, manual_step in zip(full.transitions, manual):
        _assert_transitions_equal(full_step, manual_step)
    _assert_states_equal(full.final_state, suffix.final_state)
    _assert_states_equal(full.final_state, manual[-1].state_after)
    full_metrics = _metrics(full_components.evaluator, full)
    resumed_metrics = _metrics(suffix_components.evaluator, suffix)
    assert full_metrics == resumed_metrics
    assert full_metrics.loss < initial_metrics.loss
    assert full_metrics.accuracy >= 0.95
