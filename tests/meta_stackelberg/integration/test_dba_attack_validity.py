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
from meta_stackelberg.security.attacks.dba import (
    DistributedBackdoorUpdateGenerator,
    DistributedTriggerPlan,
)
from meta_stackelberg.security.attacks.identity import IdentityMaliciousUpdateGenerator
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import CompositeTrigger, PatchTrigger
from meta_stackelberg.security.engine.attack_round_engine import AttackRoundEngine
from meta_stackelberg.security.population import FixedMaliciousPopulation
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackKnowledge


SEEDS = (601, 602, 603)
MALICIOUS_CLIENTS = frozenset({0, 1, 2})
SUB_TRIGGERS = (
    PatchTrigger(row=6, column=0, height=2, width=2, value=1.5),
    PatchTrigger(row=6, column=3, height=2, width=2, value=1.5),
    PatchTrigger(row=6, column=6, height=2, width=2, value=1.5),
)
PLAN = DistributedTriggerPlan(
    SUB_TRIGGERS,
    {client_id: client_id for client_id in MALICIOUS_CLIENTS},
    MALICIOUS_CLIENTS,
)


def _assert_trigger_plan_is_disjoint_composition() -> None:
    assert len(SUB_TRIGGERS) == 3
    assert all(isinstance(trigger, PatchTrigger) for trigger in SUB_TRIGGERS)
    full = PLAN.full_trigger
    assert isinstance(full, CompositeTrigger)
    assert len(full.triggers) == len(SUB_TRIGGERS)
    for index, component in enumerate(full.triggers):
        assert component is SUB_TRIGGERS[index]

    image = torch.zeros((1, 8, 8))
    component_results = tuple(trigger.apply(image) for trigger in SUB_TRIGGERS)
    component_masks = tuple(result.ne(image) for result in component_results)
    assert all(mask.any() for mask in component_masks)
    for index, mask in enumerate(component_masks):
        for other_mask in component_masks[index + 1:]:
            assert not torch.logical_and(mask, other_mask).any()

    expected = image.clone()
    for trigger in SUB_TRIGGERS:
        expected = trigger.apply(expected)
    actual = full.apply(image)
    assert torch.equal(actual, expected)
    assert torch.equal(actual.ne(image), torch.stack(component_masks).any(dim=0))
    for mask, component_result in zip(component_masks, component_results):
        assert torch.equal(actual[mask], component_result[mask])


def _assert_independent_train_and_held_out(
    train: TensorDataset,
    held_out: TensorDataset,
) -> None:
    assert train is not held_out
    assert len(train.tensors) == len(held_out.tensors) == 2
    for train_tensor, held_out_tensor in zip(train.tensors, held_out.tensors):
        train_storage = train_tensor.untyped_storage().data_ptr()
        held_out_storage = held_out_tensor.untyped_storage().data_ptr()
        assert train_storage != held_out_storage
    assert not torch.equal(train.tensors[0], held_out.tensors[0])


def _dataset(offset: float) -> TensorDataset:
    levels = torch.linspace(0.5 + offset, 1.0 + offset, 60)
    source = -levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    target = levels[:, None, None, None].expand(-1, 1, 8, 8).clone()
    return TensorDataset(
        torch.cat((source, target)),
        torch.cat((torch.zeros(60), torch.ones(60))).long(),
    )


def _model_factory() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    return model


@dataclass(frozen=True)
class RunResult:
    clean: ClassificationMetrics
    sub_triggers: tuple[TargetedAttackMetrics, ...]
    full_trigger: TargetedAttackMetrics
    trajectory: FederatedTrajectory


def _run(seed: int, attack: str) -> RunResult:
    train = _dataset(0.0)
    held_out = _dataset(0.1)
    _assert_independent_train_and_held_out(train, held_out)
    partitions = iid_partition(len(train), 6, RandomSource(92))
    clean_datasets = {
        client_id: Subset(train, indices)
        for client_id, indices in enumerate(partitions)
    }
    poison_fraction = 1.0 if attack == 'dba-strong' else 0.0
    poisoned_datasets = {
        client_id: SourceTargetPoisonedDataset(
            dataset=clean_datasets[client_id],
            trigger=PLAN.trigger_for(client_id),
            source_class=0,
            target_class=1,
            poison_fraction=poison_fraction,
            rng=RandomSource(20_000 + client_id),
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
        if attack == 'identity':
            generator = IdentityMaliciousUpdateGenerator(ScopedLocalTrainer(
                trainer({client_id: clean_datasets[client_id] for client_id in MALICIOUS_CLIENTS}),
                MALICIOUS_CLIENTS,
            ))
        else:
            generator = DistributedBackdoorUpdateGenerator(
                trainer=ScopedLocalTrainer(trainer(poisoned_datasets), MALICIOUS_CLIENTS),
                plan=PLAN,
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
            task_id='matched-dba-validity',
            horizon=8,
            sample_size=4,
            server_lr=1.0,
            initial_state=state,
        ),
        source,
    )
    final_state = trajectory.final_state.global_model

    def targeted(trigger) -> TargetedAttackMetrics:
        evaluator = TargetedAttackEvaluator(
            model_factory=_model_factory,
            dataset=held_out,
            codec=codec,
            trigger=trigger,
            source_class=0,
            target_class=1,
            batch_size=24,
        )
        assert evaluator.dataset is held_out
        return evaluator.evaluate(final_state)

    clean_evaluator = ClassificationEvaluator(
        model_factory=_model_factory,
        dataset=held_out,
        codec=codec,
        batch_size=24,
    )
    assert clean_evaluator.dataset is held_out

    return RunResult(
        clean=clean_evaluator.evaluate(final_state),
        sub_triggers=tuple(targeted(trigger) for trigger in SUB_TRIGGERS),
        full_trigger=targeted(PLAN.full_trigger),
        trajectory=trajectory,
    )


def _assert_snapshots_equal(left: RandomSnapshot, right: RandomSnapshot) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    assert torch.equal(left.torch_cpu_state, right.torch_cpu_state)


def test_matched_dba_preserves_clean_utility_and_shows_composite_activation() -> None:
    _assert_trigger_plan_is_disjoint_composition()
    composite_activation_gaps = []
    for seed in SEEDS:
        clean = _run(seed, 'clean')
        identity = _run(seed, 'identity')
        zero = _run(seed, 'dba-0')
        strong = _run(seed, 'dba-strong')

        assert clean.clean == identity.clean == zero.clean
        assert clean.sub_triggers == identity.sub_triggers == zero.sub_triggers
        assert clean.full_trigger == identity.full_trigger == zero.full_trigger
        for result in (clean, identity, zero, strong):
            assert len(result.sub_triggers) == len(SUB_TRIGGERS) == 3
            assert all(metric.source_examples == 60 for metric in result.sub_triggers)
            assert result.full_trigger.source_examples == 60
            assert result.clean.num_examples == 120
        assert clean.clean.accuracy == 1.0
        assert np.isfinite(strong.clean.loss)
        assert np.isfinite(strong.clean.accuracy)
        assert strong.clean.accuracy == clean.clean.accuracy
        assert strong.full_trigger.attack_success_rate > identity.full_trigger.attack_success_rate
        strongest_sub_trigger_asr = max(
            metric.attack_success_rate for metric in strong.sub_triggers
        )
        assert strong.full_trigger.attack_success_rate >= strongest_sub_trigger_asr
        composite_activation_gaps.append(
            strong.full_trigger.attack_success_rate - strongest_sub_trigger_asr
        )

        for clean_step, identity_step, zero_step, strong_step in zip(
            clean.trajectory.transitions,
            identity.trajectory.transitions,
            zero.trajectory.transitions,
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
            expected_malicious = len(set(strong_step.sampled_clients) & MALICIOUS_CLIENTS)
            for attacked_step in (identity_step, zero_step, strong_step):
                assert attacked_step.private_diagnostics['malicious_client_count'] == expected_malicious
                assert len(attacked_step.malicious_updates) == expected_malicious
            assert expected_malicious >= 1

    # Composition evidence means the full trigger activates at least as well as
    # every part on each seed, with a strict advantage somewhere in the sweep.
    assert sum(composite_activation_gaps) > 0.0
