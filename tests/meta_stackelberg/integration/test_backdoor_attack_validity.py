from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

from meta_stackelberg.core.random_state import RandomSnapshot, RandomSource
from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator, TargetedAttackMetrics
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.clients.sampling import UniformClientSampler
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.data.partitioning import iid_partition
from meta_stackelberg.federated.engine.round_engine import RoundEngine
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.episode import EpisodeRunner, EpisodeSpec, FederatedTrajectory
from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator, ClassificationMetrics
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.attacks.backdoor import BackdoorLocalUpdateGenerator
from meta_stackelberg.security.attacks.identity import IdentityMaliciousUpdateGenerator
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import PatchTrigger
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS = (401, 402, 403)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})
TRIGGER = PatchTrigger(row=6, column=6, height=2, width=2, value=5.0)


def _dataset(offset: float) -> TensorDataset:
    levels = torch.linspace(0.5 + offset, 1.0 + offset, 60)
    source = -levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    target = levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    images = torch.cat((source, target), dim=0)
    labels = torch.cat((torch.zeros(60), torch.ones(60))).long()
    return TensorDataset(images, labels)


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    return model


@dataclass(frozen=True)
class RunResult:
    clean: ClassificationMetrics
    backdoor: TargetedAttackMetrics
    trajectory: FederatedTrajectory


def _run(seed: int, attack: str) -> RunResult:
    train = _dataset(0.0)
    held_out = _dataset(0.1)
    partitions = iid_partition(len(train), 6, RandomSource(91))
    clean_datasets = {
        client_id: Subset(train, indices)
        for client_id, indices in enumerate(partitions)
    }
    poison_fraction = 1.0 if attack == 'bfl-1' else 0.0
    malicious_datasets = {
        client_id: SourceTargetPoisonedDataset(
            dataset=clean_datasets[client_id],
            trigger=TRIGGER,
            source_class=0,
            target_class=1,
            poison_fraction=poison_fraction,
            rng=RandomSource(10_000 + client_id),
        )
        for client_id in MALICIOUS_CLIENTS
    }
    codec = TorchParameterCodec()
    source = RandomSource(seed)
    state = RoundState(
        round_index=0,
        global_model=codec.capture(_model_factory()),
        random_snapshot=source.capture(),
    )

    def trainer(datasets) -> TorchLocalTrainer:
        return TorchLocalTrainer(
            model_factory=_model_factory,
            client_datasets=datasets,
            codec=codec,
            learning_rate=0.08,
            local_epochs=2,
            batch_size=8,
        )

    sampler = UniformClientSampler(num_clients=6)
    if attack == 'clean':
        engine = RoundEngine(
            sampler=sampler,
            trainer=trainer(clean_datasets),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )
    else:
        benign_ids = frozenset(set(clean_datasets) - set(MALICIOUS_CLIENTS))
        malicious_trainer = ScopedLocalTrainer(trainer(malicious_datasets), MALICIOUS_CLIENTS)
        if attack == 'identity':
            malicious_trainer = ScopedLocalTrainer(
                trainer({client_id: clean_datasets[client_id] for client_id in MALICIOUS_CLIENTS}),
                MALICIOUS_CLIENTS,
            )
            generator = IdentityMaliciousUpdateGenerator(malicious_trainer)
        else:
            generator = BackdoorLocalUpdateGenerator(
                trainer=malicious_trainer,
                source_class=0,
                target_class=1,
                poison_fraction=poison_fraction,
            )
        engine = AttackRoundEngine(
            sampler=sampler,
            benign_trainer=trainer({client_id: clean_datasets[client_id] for client_id in benign_ids}),
            malicious_generator=generator,
            population=FixedMaliciousPopulation(MALICIOUS_CLIENTS),
            knowledge=AttackKnowledge(allows_global_model=True, allows_local_data=True),
            aggregator=FedAvg(),
            server_optimizer=ServerSGD(),
        )

    trajectory = EpisodeRunner(engine).run(
        EpisodeSpec(
            task_id='matched-bfl-validity',
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    final_state = trajectory.final_state.global_model
    return RunResult(
        clean=ClassificationEvaluator(
            model_factory=_model_factory,
            dataset=held_out,
            codec=codec,
            batch_size=24,
        ).evaluate(final_state),
        backdoor=TargetedAttackEvaluator(
            model_factory=_model_factory,
            dataset=held_out,
            codec=codec,
            trigger=TRIGGER,
            source_class=0,
            target_class=1,
            batch_size=24,
        ).evaluate(final_state),
        trajectory=trajectory,
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def test_matched_bfl_poisoning_increases_held_out_asr_and_preserves_clean_utility() -> None:
    for seed in SEEDS:
        clean = _run(seed, 'clean')
        identity = _run(seed, 'identity')
        zero = _run(seed, 'bfl-0')
        strong = _run(seed, 'bfl-1')

        assert clean.clean == identity.clean == zero.clean
        assert clean.backdoor == identity.backdoor == zero.backdoor
        assert clean.clean.accuracy == 1.0
        assert strong.clean.accuracy >= 0.95
        assert strong.backdoor.attack_success_rate - identity.backdoor.attack_success_rate >= 0.8

        clean_steps = clean.trajectory.transitions
        for identity_step, zero_step, clean_step, strong_step in zip(
            identity.trajectory.transitions,
            zero.trajectory.transitions,
            clean_steps,
            strong.trajectory.transitions,
        ):
            assert clean_step.sampled_clients == identity_step.sampled_clients
            assert clean_step.sampled_clients == zero_step.sampled_clients
            assert clean_step.sampled_clients == strong_step.sampled_clients
            np.testing.assert_array_equal(clean_step.aggregate_delta.vector(), identity_step.aggregate_delta.vector())
            np.testing.assert_array_equal(clean_step.aggregate_delta.vector(), zero_step.aggregate_delta.vector())
            np.testing.assert_array_equal(clean_step.state_after.global_model.vector(), identity_step.state_after.global_model.vector())
            np.testing.assert_array_equal(clean_step.state_after.global_model.vector(), zero_step.state_after.global_model.vector())
            _assert_snapshots_equal(clean_step.state_after.random_snapshot, identity_step.state_after.random_snapshot)
            _assert_snapshots_equal(clean_step.state_after.random_snapshot, zero_step.state_after.random_snapshot)
            _assert_snapshots_equal(clean_step.state_after.random_snapshot, strong_step.state_after.random_snapshot)
            assert strong_step.private_diagnostics['malicious_client_count'] >= 1
