from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
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
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.attacks.delta_reversal import DeltaReversalAttack
from meta_stackelberg.security.attacks.identity import IdentityMaliciousUpdateGenerator
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS = (301, 302, 303)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})


def _dataset(offset_start: float) -> TensorDataset:
    offsets = torch.linspace(offset_start, offset_start + 1.0, steps=60)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = -negative
    features = torch.cat((negative, positive), dim=0)
    labels = torch.cat((
        torch.zeros(60, dtype=torch.long),
        torch.ones(60, dtype=torch.long),
    ))
    return TensorDataset(features, labels)


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


@dataclass(frozen=True)
class RunResult:
    metrics: ClassificationMetrics
    trajectory: FederatedTrajectory


def _run(seed: int, attack: str) -> RunResult:
    train_dataset = _dataset(0.0)
    held_out_dataset = _dataset(0.2)
    partitions = iid_partition(len(train_dataset), 6, RandomSource(77))
    client_datasets = {
        client_id: Subset(train_dataset, indices)
        for client_id, indices in enumerate(partitions)
    }
    codec = TorchParameterCodec()
    source = RandomSource(seed)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )

    def trainer(client_ids: frozenset[int] | None = None) -> TorchLocalTrainer:
        datasets = client_datasets
        if client_ids is not None:
            datasets = {
                client_id: client_datasets[client_id]
                for client_id in client_ids
            }
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=datasets,
            codec=codec,
            learning_rate=0.2,
            local_epochs=1,
            batch_size=8,
        )

    sampler = UniformClientSampler(num_clients=6)
    if attack == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(client_datasets) - set(MALICIOUS_CLIENTS))
        benign_trainer = trainer(benign_ids)
        malicious_trainer = ScopedLocalTrainer(
            trainer(MALICIOUS_CLIENTS),
            MALICIOUS_CLIENTS,
        )
        if attack == 'identity':
            generator = IdentityMaliciousUpdateGenerator(malicious_trainer)
        elif attack == 'delta-reversal':
            generator = DeltaReversalAttack(trainer=malicious_trainer, budget=2.0)
        else:
            raise ValueError(f'unknown test attack {attack!r}')
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=benign_trainer,
            malicious_generator=generator,
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=AttackKnowledge(
                allows_global_model=True,
                allows_local_data=True,
            ),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )
    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec(
            task_id='matched-untargeted-validity',
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    evaluator = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out_dataset,
        codec=codec,
        batch_size=24,
    )
    return RunResult(
        metrics=evaluator.evaluate(trajectory.final_state.global_model),
        trajectory=trajectory,
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def test_matched_identity_is_exact_and_delta_reversal_causes_held_out_harm() -> None:
    loss_harms = []
    for seed in SEEDS:
        clean = _run(seed, 'clean')
        identity = _run(seed, 'identity')
        attacked = _run(seed, 'delta-reversal')

        assert clean.metrics == identity.metrics
        assert clean.metrics.accuracy == 1.0
        assert [step.sampled_clients for step in clean.trajectory.transitions] == [
            step.sampled_clients for step in identity.trajectory.transitions
        ]
        assert [step.sampled_clients for step in clean.trajectory.transitions] == [
            step.sampled_clients for step in attacked.trajectory.transitions
        ]
        for clean_step, identity_step, attacked_step in zip(
            clean.trajectory.transitions,
            identity.trajectory.transitions,
            attacked.trajectory.transitions,
        ):
            np.testing.assert_array_equal(
                clean_step.aggregate_delta.vector(),
                identity_step.aggregate_delta.vector(),
            )
            np.testing.assert_array_equal(
                clean_step.state_after.global_model.vector(),
                identity_step.state_after.global_model.vector(),
            )
            _assert_snapshots_equal(
                clean_step.state_after.random_snapshot,
                identity_step.state_after.random_snapshot,
            )
            _assert_snapshots_equal(
                clean_step.state_after.random_snapshot,
                attacked_step.state_after.random_snapshot,
            )
            assert attacked_step.private_diagnostics['malicious_client_count'] >= 1
            assert 'malicious_client_ids' not in attacked_step.public_signals

        loss_harm = attacked.metrics.loss - clean.metrics.loss
        accuracy_harm = clean.metrics.accuracy - attacked.metrics.accuracy
        assert loss_harm > 0.1
        assert accuracy_harm >= 0.0
        loss_harms.append(loss_harm)

    assert len(loss_harms) == len(SEEDS)
    assert all(harm > 0.0 for harm in loss_harms)
