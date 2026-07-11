from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.federated.aggregation.coordinate_median import CoordinateMedian
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec, FederatedTrajectory
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator, ClassificationMetrics
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.security.attacks._base import (
    LOCAL_MODEL_CAPABILITIES,
    as_malicious_update,
    train_base_update,
)
from meta_stackelberg.security.attacks.ipm import IPMAttack
from meta_stackelberg.security.attacks.lmp import LMPAttack
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackContext, AttackKnowledge, RoundAttackContext


SEEDS = (501, 502, 503)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})


def _dataset(offset_start: float) -> TensorDataset:
    offsets = torch.linspace(offset_start, offset_start + 1.0, steps=60)
    negative = torch.stack((-1.0 - offsets, -0.5 - 0.25 * offsets), dim=1)
    positive = -negative
    features = torch.cat((negative, positive), dim=0)
    labels = torch.cat((torch.zeros(60), torch.ones(60))).long()
    return TensorDataset(features, labels)


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


class RoundIdentityGenerator:
    capabilities = LOCAL_MODEL_CAPABILITIES

    def __init__(self, trainer: ScopedLocalTrainer) -> None:
        self.trainer = trainer
        self.allowed_client_ids = trainer.allowed_client_ids

    def craft_round(
        self,
        context: RoundAttackContext,
        rngs: tuple[RandomSource, ...],
    ) -> tuple[ClientUpdate, ...]:
        return tuple(
            as_malicious_update(train_base_update(
                self.trainer,
                AttackContext(
                    client_id=client_id,
                    round_index=context.round_index,
                    global_model=context.global_model,
                ),
                rng,
            ))
            for client_id, rng in zip(context.malicious_client_ids, rngs)
        )


@dataclass(frozen=True)
class RunResult:
    metrics: ClassificationMetrics
    trajectory: FederatedTrajectory


def _run(seed: int, attack: str, aggregation: str) -> RunResult:
    train_dataset = _dataset(0.0)
    held_out = _dataset(0.2)
    partitions = iid_partition(len(train_dataset), 6, RandomSource(77))
    datasets = {
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

    def trainer(client_ids=None) -> TorchLocalTrainer:
        selected = datasets if client_ids is None else {
            client_id: datasets[client_id] for client_id in client_ids
        }
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=selected,
            codec=codec,
            learning_rate=0.2,
            local_epochs=1,
            batch_size=8,
        )

    aggregator = FedAvg() if aggregation == 'fedavg' else CoordinateMedian()
    sampler = UniformClientSampler(num_clients=6)
    if attack == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(),
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(datasets) - set(MALICIOUS_CLIENTS))
        if attack == 'identity':
            generator = RoundIdentityGenerator(ScopedLocalTrainer(
                trainer(MALICIOUS_CLIENTS), MALICIOUS_CLIENTS
            ))
            knowledge = AttackKnowledge(allows_global_model=True, allows_local_data=True)
        elif attack == 'ipm':
            generator = IPMAttack(
                scale=3.0,
                num_examples_by_client={client_id: len(datasets[client_id]) for client_id in MALICIOUS_CLIENTS},
            )
            knowledge = AttackKnowledge(allows_benign_updates=True)
        elif attack == 'lmp':
            generator = LMPAttack(
                scale=5.0,
                num_examples_by_client={client_id: len(datasets[client_id]) for client_id in MALICIOUS_CLIENTS},
            )
            knowledge = AttackKnowledge(
                allows_global_model=True,
                allows_benign_updates=True,
            )
        else:
            raise ValueError(attack)
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer(benign_ids),
            malicious_generator=generator,
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=knowledge,
            aggregator=aggregator,
            server_optimizer=ServerSGD(),
        )
    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec(
            task_id=f'matched-{attack}-{aggregation}',
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    metrics = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out,
        codec=codec,
        batch_size=24,
    ).evaluate(trajectory.final_state.global_model)
    return RunResult(metrics=metrics, trajectory=trajectory)


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def _assert_identity_exact(clean: RunResult, identity: RunResult) -> None:
    assert clean.metrics == identity.metrics
    for clean_step, identity_step in zip(clean.trajectory.transitions, identity.trajectory.transitions):
        assert clean_step.sampled_clients == identity_step.sampled_clients
        np.testing.assert_array_equal(clean_step.aggregate_delta.vector(), identity_step.aggregate_delta.vector())
        np.testing.assert_array_equal(clean_step.state_after.global_model.vector(), identity_step.state_after.global_model.vector())
        _assert_snapshots_equal(clean_step.state_after.random_snapshot, identity_step.state_after.random_snapshot)


def test_ipm_has_negative_inner_product_and_harms_fedavg_across_seeds() -> None:
    for seed in SEEDS:
        clean = _run(seed, 'clean', 'fedavg')
        identity = _run(seed, 'identity', 'fedavg')
        attacked = _run(seed, 'ipm', 'fedavg')
        _assert_identity_exact(clean, identity)
        assert clean.metrics.accuracy == identity.metrics.accuracy == 1.0
        assert np.isfinite(attacked.metrics.loss)
        assert np.isfinite(attacked.metrics.accuracy)
        assert attacked.metrics.loss > identity.metrics.loss + 0.1
        for identity_step, step in zip(identity.trajectory.transitions, attacked.trajectory.transitions):
            assert step.private_diagnostics['malicious_client_count'] >= 1
            assert step.private_diagnostics['malicious_client_count'] == len(step.malicious_updates)
            _assert_snapshots_equal(
                identity_step.state_after.random_snapshot,
                step.state_after.random_snapshot,
            )
            benign_mean = np.mean([update.delta.vector() for update in step.benign_updates], axis=0)
            for malicious in step.malicious_updates:
                assert np.isfinite(np.linalg.norm(malicious.delta.vector()))
                assert float(np.dot(malicious.delta.vector(), benign_mean)) < 0.0


def test_lmp_changes_coordinate_median_outcome_with_matched_sampling() -> None:
    for seed in SEEDS:
        clean = _run(seed, 'clean', 'median')
        identity = _run(seed, 'identity', 'median')
        attacked = _run(seed, 'lmp', 'median')
        _assert_identity_exact(clean, identity)
        assert clean.metrics.accuracy == identity.metrics.accuracy == 1.0
        assert np.isfinite(attacked.metrics.loss)
        assert np.isfinite(attacked.metrics.accuracy)
        assert attacked.metrics.loss > identity.metrics.loss + 0.01
        assert [step.sampled_clients for step in identity.trajectory.transitions] == [
            step.sampled_clients for step in attacked.trajectory.transitions
        ]
        for identity_step, attacked_step in zip(identity.trajectory.transitions, attacked.trajectory.transitions):
            _assert_snapshots_equal(
                identity_step.state_after.random_snapshot,
                attacked_step.state_after.random_snapshot,
            )
            assert attacked_step.private_diagnostics['malicious_client_count'] >= 1
            assert attacked_step.private_diagnostics['malicious_client_count'] == len(
                attacked_step.malicious_updates
            )
            for malicious in attacked_step.malicious_updates:
                assert np.isfinite(np.linalg.norm(malicious.delta.vector()))
                assert malicious.metadata['variant'] == 'median_craft_real'
