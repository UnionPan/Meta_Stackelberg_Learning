import copy
from dataclasses import FrozenInstanceError
from dataclasses import dataclass
import pickle

import numpy as np
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
    return _plan_with_triggers(_triggers())


def _plan_with_triggers(triggers):
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    return DistributedTriggerPlan(triggers, {3: 0, 7: 1}, {3, 7})


def _assert_rng_snapshot_equal(left, right) -> None:
    assert left.python_state == right.python_state
    assert left.numpy_state == right.numpy_state
    torch.testing.assert_close(left.torch_cpu_state, right.torch_cpu_state, rtol=0, atol=0)


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

    with pytest.raises(TypeError, match='scalar equality.*value-distinct'):
        DistributedTriggerPlan(
            (
                TensorTrigger(torch.tensor([1.0, 2.0])),
                TensorTrigger(torch.tensor([1.0, 2.0])),
            ),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_rejects_protocol_trigger_with_ambiguous_equality() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    @dataclass(frozen=True, eq=False)
    class AmbiguousEqualityTrigger:
        value: int

        def __eq__(self, other: object) -> torch.Tensor:
            return torch.tensor([True, True])

        def __hash__(self) -> int:
            return hash(self.value)

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    with pytest.raises(TypeError, match='scalar equality.*value-distinct'):
        DistributedTriggerPlan(
            (AmbiguousEqualityTrigger(1), AmbiguousEqualityTrigger(1)),
            {3: 0, 7: 1},
            {3, 7},
        )


def test_distributed_trigger_plan_accepts_plain_protocol_triggers() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class PlainTrigger:
        def __init__(self, value: float) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, PlainTrigger) and self.value == other.value

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    plan = DistributedTriggerPlan(
        (PlainTrigger(1.0), PlainTrigger(2.0)),
        {3: 0, 7: 1},
        {3, 7},
    )

    assert plan.trigger_for(3) == PlainTrigger(1.0)


def test_client_assignment_property_returns_stable_read_only_mapping() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    plan = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})
    first_view = plan.client_to_sub_trigger
    second_view = plan.client_to_sub_trigger

    assert first_view is second_view
    assert dict(first_view) == {3: 0, 7: 1}
    with pytest.raises(TypeError):
        first_view[3] = 1


def test_distributed_trigger_plan_has_stable_value_semantics() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    first = DistributedTriggerPlan(_triggers(), {7: 1, 3: 0}, {7, 3})
    second = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})

    assert first == second


@pytest.mark.parametrize(
    'roundtrip',
    [copy.deepcopy, lambda plan: pickle.loads(pickle.dumps(plan))],
)
def test_distributed_trigger_plan_roundtrips_with_stable_read_only_mapping(roundtrip) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    plan = DistributedTriggerPlan(_triggers(), {3: 0, 7: 1}, {3, 7})
    rebuilt = roundtrip(plan)
    first_view = rebuilt.client_to_sub_trigger

    assert rebuilt == plan
    assert first_view is rebuilt.client_to_sub_trigger
    assert dict(first_view) == {3: 0, 7: 1}
    with pytest.raises(TypeError):
        first_view[3] = 1


def test_distributed_trigger_plan_wraps_unexpected_equality_errors() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class BrokenEqualityTrigger:
        def __eq__(self, other: object) -> bool:
            raise LookupError('implementation detail')

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return image.clone()

    with pytest.raises(TypeError, match='scalar equality.*value-distinct'):
        DistributedTriggerPlan(
            (BrokenEqualityTrigger(), BrokenEqualityTrigger()),
            {3: 0, 7: 1},
            {3, 7},
        )


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

    triggers = _triggers()
    trainer = _distributed_trainer(triggers=triggers)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan_with_triggers(triggers),
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


def test_dba_generator_requires_dataset_trigger_identity_even_when_equal() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator
    from meta_stackelberg.security.attacks.dba import DistributedTriggerPlan

    class EqualButDifferentTrigger:
        def __init__(self, equality_key: int, fill: float) -> None:
            self.equality_key = equality_key
            self.fill = fill

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, EqualButDifferentTrigger)
                and self.equality_key == other.equality_key
            )

        def apply(self, image: torch.Tensor) -> torch.Tensor:
            return torch.full_like(image, self.fill)

    dataset_triggers = (
        EqualButDifferentTrigger(1, 3.0),
        EqualButDifferentTrigger(2, 7.0),
    )
    plan_triggers = (
        EqualButDifferentTrigger(1, 30.0),
        EqualButDifferentTrigger(2, 70.0),
    )
    trainer = _distributed_trainer(triggers=dataset_triggers)
    plan = DistributedTriggerPlan(plan_triggers, {3: 0, 7: 1}, {3, 7})

    with pytest.raises(ValueError, match='trigger'):
        DistributedBackdoorUpdateGenerator(
            trainer=trainer,
            plan=plan,
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


def test_dba_generator_preserves_exact_trainer_base_trainer_and_datasets() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    triggers = _triggers()
    trainer = _distributed_trainer(triggers=triggers)
    plan = _plan_with_triggers(triggers)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=plan,
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )

    assert generator.trainer is trainer
    assert generator.trainer.base_trainer is trainer.base_trainer
    assert generator.allowed_client_ids == frozenset({3, 7})
    assert generator.trainer.dataset_for(3) is trainer.dataset_for(3)
    assert generator.trainer.dataset_for(7) is trainer.dataset_for(7)


def test_dba_generator_accepts_non_deepcopyable_dataset_and_crafts_real_update() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class NonDeepcopyableTensorDataset(TensorDataset):
        def __deepcopy__(self, memo):
            del memo
            raise TypeError('must remain shared')

    triggers = _triggers()
    clean = NonDeepcopyableTensorDataset(
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
            poison_fraction=0.5,
            rng=RandomSource(client_id),
        )
        for client_id, trigger in zip((3, 7), triggers, strict=True)
    }
    trainer = ScopedLocalTrainer(
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
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan_with_triggers(triggers),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    context = AttackContext(
        client_id=7,
        round_index=4,
        global_model=TorchParameterCodec().capture(_model()),
    )

    update = generator.craft(context, RandomSource(31))

    assert update.num_examples == 4
    assert update.is_malicious
    assert torch.count_nonzero(torch.from_numpy(update.delta.vector())).item() > 0


def test_dba_generator_requires_exact_builtin_poisoned_dataset() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class DerivedPoisonedDataset(SourceTargetPoisonedDataset):
        pass

    triggers = _triggers()
    trainer = _distributed_trainer(triggers=triggers)
    original = trainer.dataset_for(3)
    derived = DerivedPoisonedDataset.__new__(DerivedPoisonedDataset)
    derived.__dict__.update(original.__dict__)
    trainer.base_trainer.client_datasets[3] = derived

    with pytest.raises(TypeError, match='exact.*SourceTargetPoisonedDataset'):
        DistributedBackdoorUpdateGenerator(
            trainer=trainer,
            plan=_plan_with_triggers(triggers),
            source_class=0,
            target_class=1,
            poison_fraction=0.5,
        )


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


@pytest.mark.parametrize('poison_fraction', [np.bool_(False), np.bool_(True), '0.5', np.array(0.5)])
def test_dba_generator_rejects_non_real_scalar_poison_fraction(poison_fraction) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    with pytest.raises(TypeError, match='poison_fraction'):
        DistributedBackdoorUpdateGenerator(
            trainer=_distributed_trainer(),
            plan=_plan(),
            source_class=0,
            target_class=1,
            poison_fraction=poison_fraction,
        )


def test_dba_generator_accepts_numpy_real_scalar_poison_fraction() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    triggers = _triggers()
    generator = DistributedBackdoorUpdateGenerator(
        trainer=_distributed_trainer(triggers=triggers),
        plan=_plan_with_triggers(triggers),
        source_class=0,
        target_class=1,
        poison_fraction=np.float32(0.5),
    )

    assert generator.poison_fraction == 0.5


@pytest.mark.parametrize(
    'mutate',
    [
        lambda trainer: setattr(trainer.dataset_for(7), 'trigger', _triggers()[0]),
        lambda trainer: setattr(trainer.dataset_for(7), 'source_class', 1),
        lambda trainer: setattr(trainer.dataset_for(7), 'target_class', 0),
        lambda trainer: setattr(trainer.dataset_for(7), 'poison_fraction', 1.0),
        lambda trainer: setattr(trainer.dataset_for(7), 'dataset', TensorDataset(
            torch.zeros((1, 1, 2, 2)), torch.zeros(1, dtype=torch.long)
        )),
        lambda trainer: trainer.base_trainer.client_datasets.__setitem__(
            7,
            TensorDataset(torch.zeros((1, 1, 2, 2)), torch.zeros(1, dtype=torch.long)),
        ),
    ],
)
def test_dba_generator_revalidates_live_dataset_before_training_without_consuming_rng(
    mutate,
) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    triggers = _triggers()
    trainer = _distributed_trainer(triggers=triggers)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan_with_triggers(triggers),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    mutate(trainer)
    rng = RandomSource(41)
    before = rng.capture()

    with pytest.raises((TypeError, ValueError), match='dataset|trigger|configuration'):
        generator.craft(
            AttackContext(
                client_id=7,
                round_index=4,
                global_model=TorchParameterCodec().capture(_model()),
            ),
            rng,
        )

    _assert_rng_snapshot_equal(rng.capture(), before)


@pytest.mark.parametrize(
    'mutate',
    [
        lambda dataset: setattr(dataset, 'poisoned_indices', frozenset()),
        lambda dataset: setattr(dataset, 'eligible_count', dataset.eligible_count + 1),
    ],
)
def test_dba_generator_rejects_poison_view_state_mutation_without_consuming_rng(
    mutate,
) -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    triggers = _triggers()
    trainer = _distributed_trainer(triggers=triggers)
    generator = DistributedBackdoorUpdateGenerator(
        trainer=trainer,
        plan=_plan_with_triggers(triggers),
        source_class=0,
        target_class=1,
        poison_fraction=0.5,
    )
    mutate(trainer.dataset_for(7))
    rng = RandomSource(43)
    before = rng.capture()

    with pytest.raises(ValueError, match='poison view state'):
        generator.craft(
            AttackContext(
                client_id=7,
                round_index=4,
                global_model=TorchParameterCodec().capture(_model()),
            ),
            rng,
        )

    _assert_rng_snapshot_equal(rng.capture(), before)


def test_dba_generator_requires_exact_builtin_torch_trainer() -> None:
    from meta_stackelberg.security.attacks.dba import DistributedBackdoorUpdateGenerator

    class DerivedTorchLocalTrainer(TorchLocalTrainer):
        pass

    base = _distributed_trainer().base_trainer
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
            trainer=ScopedLocalTrainer(_distributed_trainer().base_trainer, {client_id}),
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
