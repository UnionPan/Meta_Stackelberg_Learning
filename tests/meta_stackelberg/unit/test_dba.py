import copy
from dataclasses import FrozenInstanceError
from dataclasses import dataclass
import pickle

import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import CompositeTrigger, PatchTrigger
from meta_stackelberg.security.protocols import MaliciousUpdateGenerator
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackContext


def _triggers():
    return (
        PatchTrigger(row=0, column=0, height=1, width=1, value=1.0),
        PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
    )


def _model() -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    return model


def _distributed_trainer(*, triggers=None, poison_fraction=0.5) -> ScopedLocalTrainer:
    triggers = _triggers() if triggers is None else triggers
    clean = TensorDataset(
        torch.tensor([
            [[[-1.0, -1.0], [-1.0, -1.0]]],
            [[[-0.5, -0.5], [-0.5, -0.5]]],
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]),
        torch.tensor([0, 0, 1, 1]),
    )
    datasets = {
        client_id: SourceTargetPoisonedDataset(
            dataset=clean,
            trigger=trigger,
            source_class=0,
            target_class=1,
            poison_fraction=poison_fraction,
            rng=RandomSource(client_id),
        )
        for client_id, trigger in zip((3, 7), triggers, strict=True)
    }
    return ScopedLocalTrainer(
        TorchLocalTrainer(
            model_factory=_model,
            client_datasets=datasets,
            codec=TorchParameterCodec(),
            learning_rate=0.1,
            local_epochs=1,
            batch_size=2,
        ),
        {3, 7},
    )


def _plan():
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    return DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})


def test_distributed_trigger_plan_freezes_inputs_and_resolves_triggers() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    sub_triggers = list(_triggers())
    assignments = {3: 0, 7: 1}
    plan = DistributedTriggerPlan(sub_triggers, assignments, {3, 7})
    sub_triggers.clear()
    assignments[3] = 1

    assert plan.sub_triggers == _triggers()
    assert dict(plan.client_to_sub_trigger) == {3: 0, 7: 1}
    assert plan.trigger_for(3) is plan.sub_triggers[0]
    assert plan.trigger_for(7) is plan.sub_triggers[1]
    assert plan.full_trigger == CompositeTrigger(plan.sub_triggers)
    with pytest.raises(TypeError):
        plan.client_to_sub_trigger[3] = 1
    with pytest.raises(FrozenInstanceError):
        plan.sub_triggers = ()


@pytest.mark.parametrize('client_id', [True, 1.5, '1', -1])
def test_distributed_trigger_plan_rejects_invalid_client_ids(client_id) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    error = TypeError if client_id != -1 else ValueError
    with pytest.raises(error):
        DistributedTriggerPlan(_triggers(), {client_id: 0}, {client_id})


@pytest.mark.parametrize('index', [True, 0.5, '0', -1, 2])
def test_distributed_trigger_plan_rejects_invalid_sub_trigger_indices(index) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    error = TypeError if index not in (-1, 2) else ValueError
    with pytest.raises(error):
        DistributedTriggerPlan(_triggers(), {3: index, 7: 1}, {3, 7})


def test_distributed_trigger_plan_requires_exact_attacker_scope_and_uses_every_trigger() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    with pytest.raises(ValueError, match='unassigned'):
        DistributedTriggerPlan(_triggers(), {3: 0}, {3, 7})
    with pytest.raises(ValueError, match='outside'):
        DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3})
    with pytest.raises(ValueError, match='unused'):
        DistributedTriggerPlan(_triggers(), {3: 0, 7: 0}, {3, 7})


def test_distributed_trigger_plan_requires_two_value_distinct_sub_triggers() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    first = PatchTrigger(row=0, column=0, height=1, width=1, value=1.0)
    with pytest.raises(ValueError, match='at least two'):
        DistributedTriggerPlan((first,), {3: 0}, {3})
    with pytest.raises(ValueError, match='distinct'):
        DistributedTriggerPlan(
            (first, PatchTrigger(row=0, column=0, height=1, width=1, value=1.0)),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_non_scalar_trigger_equality() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass(frozen=True)
    class TensorTrigger:
        values: torch.Tensor

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    with pytest.raises(TypeError, match='deeply immutable'):
        DistributedTriggerPlan(
            (
                TensorTrigger(torch.tensor([1.0, 2.0])),
                TensorTrigger(torch.tensor([1.0, 2.0])),
            ),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_mutable_dataclass_trigger() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass
    class MutableTrigger:
        value: float

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    with pytest.raises(TypeError, match='frozen dataclass'):
        DistributedTriggerPlan(
            (MutableTrigger(1.0), MutableTrigger(2.0)),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_undecorated_dataclass_subclass() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class StatefulPatch(PatchTrigger):
        pass

    first = StatefulPatch(row=0, column=0, height=1, width=1, value=1.0)
    second = StatefulPatch(row=1, column=1, height=1, width=1, value=1.0)
    first.override = 7.0
    assert first.override == 7.0

    with pytest.raises(TypeError, match='directly decorated'):
        DistributedTriggerPlan((first, second), {3: 0, 7: 1}, {3, 7})


def test_distributed_trigger_plan_accepts_direct_frozen_dataclass_trigger() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass(frozen=True)
    class FrozenTrigger:
        value: int

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    plan = DistributedTriggerPlan(
        (FrozenTrigger(1), FrozenTrigger(2)),
        {3: 0, 7: 1},
        {3, 7},
    )

    assert plan.trigger_for(3) == FrozenTrigger(1)


def test_distributed_trigger_plan_rejects_mutable_hashable_nested_state() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class MutableHashableState:
        def __init__(self, value: int) -> None:
            self.value = value

        def __hash__(self) -> int:
            return hash(self.value)

    @dataclass(frozen=True)
    class NestedStateTrigger:
        state: MutableHashableState

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    with pytest.raises(TypeError, match='deeply immutable'):
        DistributedTriggerPlan(
            (
                NestedStateTrigger(MutableHashableState(1)),
                NestedStateTrigger(MutableHashableState(2)),
            ),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_stateful_tuple_subclass() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class StatefulTuple(tuple):
        pass

    @dataclass(frozen=True)
    class TupleTrigger:
        values: tuple[int, ...]

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    first_values = StatefulTuple((1, 2))
    first_values.override = 7
    assert first_values.override == 7

    with pytest.raises(TypeError, match='deeply immutable'):
        DistributedTriggerPlan(
            (TupleTrigger(first_values), TupleTrigger(StatefulTuple((3, 4)))),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_composite_with_unhashable_component() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass(frozen=True)
    class UnhashableTrigger:
        values: list[float]

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    composite = CompositeTrigger((UnhashableTrigger([1.0]),))
    with pytest.raises(TypeError, match='deeply immutable'):
        DistributedTriggerPlan((composite, _triggers()[0]), {3: 0, 7: 1}, {3, 7})


def test_distributed_trigger_plan_accepts_deeply_immutable_composite() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    composite = CompositeTrigger((_triggers()[0],))
    plan = DistributedTriggerPlan((composite, _triggers()[1]), {3: 0, 7: 1}, {3, 7})

    assert plan.trigger_for(3) == composite


def test_distributed_trigger_plan_rejects_cyclic_frozen_dataclass_state() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass(frozen=True)
    class CyclicTrigger:
        nested: object = None

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    cyclic = CyclicTrigger()
    object.__setattr__(cyclic, 'nested', cyclic)
    with pytest.raises(TypeError, match='acyclic'):
        DistributedTriggerPlan((cyclic, _triggers()[0]), {3: 0, 7: 1}, {3, 7})


def test_distributed_trigger_plan_has_stable_value_semantics_and_roundtrips() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    first = DistributedTriggerPlan(_triggers(), {7: 1, 3: 0}, {7, 3})
    second = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})

    assert first == second
    assert hash(first) == hash(second)
    assert pickle.loads(pickle.dumps(first)) == first
    assert copy.deepcopy(first) == first


def test_client_assignment_property_returns_detached_read_only_views() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    plan = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})
    first_view = plan.client_to_sub_trigger
    second_view = plan.client_to_sub_trigger

    assert first_view is not second_view
    assert dict(first_view) == {3: 0, 7: 1}
    with pytest.raises(TypeError):
        first_view[3] = 1


@pytest.mark.parametrize('client_id', [True, 1.5, '3', -1])
def test_trigger_for_rejects_invalid_client_id(client_id) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    plan = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})
    error = TypeError if client_id != -1 else ValueError
    with pytest.raises(error):
        plan.trigger_for(client_id)


def test_trigger_for_rejects_client_outside_attacker_scope() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    plan = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})
    with pytest.raises(KeyError, match='not assigned'):
        plan.trigger_for(9)


def test_dba_generator_preserves_real_local_update_and_adds_assignment_metadata() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    trainer = _distributed_trainer()
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan(),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    global_model = TorchParameterCodec().capture(_model())
    context = AttackContext(client_id=7, round_index=4, global_model=global_model)
    expected = trainer.train(
        7,
        RoundState(
            round_index=4,
            global_model=global_model,
            random_snapshot=RandomSource(0).capture(),
        ),
        RandomSource(19),
    )

    update = generator.craft(context, RandomSource(19))

    assert generator.capabilities.needs_global_model
    assert generator.capabilities.needs_local_data
    assert generator.allowed_client_ids == frozenset({3, 7})
    assert isinstance(generator, MaliciousUpdateGenerator)
    assert update.client_id == 7
    assert update.num_examples == expected.num_examples == 4
    assert update.is_malicious
    assert torch.count_nonzero(torch.from_numpy(expected.delta.vector())).item() > 0
    torch.testing.assert_close(
        torch.from_numpy(update.delta.vector()),
        torch.from_numpy(expected.delta.vector()),
        rtol=0,
        atol=0,
    )
    assert update.metadata == {
        **expected.metadata,
        'attack_type': 'dba',
        'source_class': 0,
        'target_class': 1,
        'poison_fraction': 0.5,
        'sub_trigger_index': 1,
        'sub_trigger_count': 2,
    }


def test_dba_generator_uses_trigger_value_semantics_and_freezes_scope() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    trainer = _distributed_trainer(triggers=tuple(copy.deepcopy(trigger) for trigger in _triggers()))
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan(),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    trainer._trainer.client_datasets.clear()

    assert generator.allowed_client_ids == frozenset({3, 7})
    assert generator.trainer.dataset_for(3) is not None


def test_dba_generator_snapshots_poison_views_against_external_mutation() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    trainer = _distributed_trainer()
    original = trainer.dataset_for(7)
    original_indices = frozenset(original.poisoned_indices)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan(),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    context = AttackContext(
        client_id=7,
        round_index=4,
        global_model=TorchParameterCodec().capture(_model()),
    )
    before = generator.craft(context, RandomSource(23))

    original.trigger = _triggers()[0]
    original.target_class = 0
    original.poison_fraction = 1.0
    original.eligible_count = 99
    original.poisoned_indices = frozenset()
    after = generator.craft(context, RandomSource(23))
    owned = generator.trainer.dataset_for(7)

    assert owned is not original
    assert owned.trigger == _triggers()[1]
    assert owned.source_class == 0
    assert owned.target_class == 1
    assert owned.poison_fraction == 0.5
    assert owned.eligible_count == 2
    assert owned.poisoned_indices == original_indices
    torch.testing.assert_close(
        torch.from_numpy(after.delta.vector()),
        torch.from_numpy(before.delta.vector()),
        rtol=0,
        atol=0,
    )


def test_dba_generator_snapshots_underlying_dataset_against_tensor_mutation() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    trainer = _distributed_trainer()
    original_view = trainer.dataset_for(7)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan(),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    owned_view = generator.trainer.dataset_for(7)
    context = AttackContext(
        client_id=7,
        round_index=4,
        global_model=TorchParameterCodec().capture(_model()),
    )
    owned_image_before = owned_view.dataset[0][0].clone()
    update_before = generator.craft(context, RandomSource(29))

    original_view.dataset.tensors[0].add_(1000.0)
    update_after = generator.craft(context, RandomSource(29))

    assert owned_view.dataset is not original_view.dataset
    assert generator.trainer.dataset_for(3).dataset is owned_view.dataset
    torch.testing.assert_close(owned_view.dataset[0][0], owned_image_before, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.from_numpy(update_after.delta.vector()),
        torch.from_numpy(update_before.delta.vector()),
        rtol=0,
        atol=0,
    )


def test_dba_generator_reports_non_scalar_dataset_trigger_equality() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class NonScalarEqualityTrigger:
        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

        def __eq__(self, other):
            del other
            return torch.tensor([True, False])

    trainer = _distributed_trainer()
    trainer.dataset_for(3).trigger = NonScalarEqualityTrigger()

    with pytest.raises(TypeError, match='scalar equality'):
        DistributedBackdoorUpdateGenerator(
            trainer=trainer,
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


def test_dba_generator_rejects_poisoned_dataset_subclass_that_aliases_copy() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class AliasingPoisonedDataset(SourceTargetPoisonedDataset):
        def __copy__(self):
            return self

    trainer = _distributed_trainer()
    original = trainer.dataset_for(3)
    aliasing = AliasingPoisonedDataset.__new__(AliasingPoisonedDataset)
    aliasing.__dict__.update(original.__dict__)
    trainer._trainer.client_datasets[3] = aliasing

    with pytest.raises(TypeError, match='exact.*SourceTargetPoisonedDataset'):
        DistributedBackdoorUpdateGenerator(
            trainer=trainer,
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


def test_dba_generator_wraps_dataset_trigger_equality_runtime_error() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class RaisingEqualityTrigger:
        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

        def __eq__(self, other):
            del other
            raise RuntimeError('equality exploded')

    trainer = _distributed_trainer()
    trainer.dataset_for(3).trigger = RaisingEqualityTrigger()

    with pytest.raises(TypeError, match='scalar equality') as caught:
        DistributedBackdoorUpdateGenerator(
            trainer=trainer,
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ('source_class', 'target_class', 'poison_fraction'),
    [(1, 1, 0.5), (0, 1, -0.1), (0, 1, 1.1), (0, 1, float('nan'))],
)
def test_dba_generator_reuses_bfl_label_and_fraction_validation(
    source_class, target_class, poison_fraction
) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    with pytest.raises(ValueError):
        DistributedBackdoorUpdateGenerator(
            trainer=_distributed_trainer(),
            plan=_plan(),
            source_class=source_class,
            target_class=target_class,
            poison_fraction=poison_fraction,
        )


@pytest.mark.parametrize('poison_fraction', [False, True])
def test_dba_generator_rejects_bool_poison_fraction(poison_fraction) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    with pytest.raises(TypeError, match='poison_fraction'):
        DistributedBackdoorUpdateGenerator(
            trainer=_distributed_trainer(poison_fraction=float(poison_fraction)),
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=poison_fraction,
        )


def test_dba_generator_requires_exact_builtin_torch_trainer() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class DerivedTorchLocalTrainer(TorchLocalTrainer):
        pass

    base = _distributed_trainer()._trainer
    derived = DerivedTorchLocalTrainer(
        model_factory=base.model_factory,
        client_datasets=base.client_datasets,
        codec=base.codec,
        learning_rate=base.learning_rate,
        local_epochs=base.local_epochs,
        batch_size=base.batch_size,
    )
    with pytest.raises(TypeError, match='exact.*TorchLocalTrainer'):
        DistributedBackdoorUpdateGenerator(
            trainer=ScopedLocalTrainer(derived, {3, 7}),
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


@pytest.mark.parametrize('client_id', [True, 9])
def test_dba_generator_rejects_invalid_or_unplanned_scoped_client(client_id) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    with pytest.raises((TypeError, ValueError)):
        DistributedBackdoorUpdateGenerator(
            trainer=ScopedLocalTrainer(_distributed_trainer()._trainer, {client_id}),
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


def test_dba_generator_rejects_dataset_configuration_or_trigger_mismatch() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    mismatched = _distributed_trainer(triggers=(_triggers()[1], _triggers()[0]))
    with pytest.raises(ValueError, match='trigger'):
        DistributedBackdoorUpdateGenerator(
            trainer=mismatched,
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )
