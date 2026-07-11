"""Distributed trigger assignments for distributed backdoor attacks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks._base import (
    LOCAL_MODEL_CAPABILITIES,
    as_malicious_update,
    train_base_update,
)
from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import CompositeTrigger, ImageTrigger
from meta_stackelberg.security.training import ScopedLocalTrainer
from meta_stackelberg.security.types import AttackContext


@dataclass(frozen=True, init=False)
class DistributedTriggerPlan:
    """An immutable assignment of distinct trigger parts to scoped attackers."""

    sub_triggers: tuple[ImageTrigger, ...]
    _client_to_sub_trigger: Mapping[int, int]
    attacker_client_ids: frozenset[int]

    def __init__(
        self,
        sub_triggers: Sequence[ImageTrigger],
        client_to_sub_trigger: Mapping[int, int],
        attacker_client_ids: Iterable[int],
    ) -> None:
        triggers = tuple(sub_triggers)
        if len(triggers) < 2:
            raise ValueError('distributed trigger plan requires at least two sub-triggers')
        if any(not isinstance(trigger, ImageTrigger) for trigger in triggers):
            raise TypeError('sub-triggers must satisfy ImageTrigger')
        if any(
            _triggers_equal(left, right)
            for index, left in enumerate(triggers)
            for right in triggers[index + 1 :]
        ):
            raise ValueError('sub-triggers must be distinct')

        assignments = dict(client_to_sub_trigger)
        _validate_client_ids(assignments)
        _validate_indices(assignments.values(), len(triggers))

        attacker_values = tuple(attacker_client_ids)
        _validate_client_ids(attacker_values)
        attackers = frozenset(int(client_id) for client_id in attacker_values)
        assigned_clients = frozenset(int(client_id) for client_id in assignments)
        if attackers - assigned_clients:
            raise ValueError('scoped attackers must not be unassigned')
        if assigned_clients - attackers:
            raise ValueError('trigger assignments must not include clients outside attacker scope')
        if set(assignments.values()) != set(range(len(triggers))):
            raise ValueError('distributed trigger plan contains unused sub-triggers')

        object.__setattr__(self, 'sub_triggers', triggers)
        object.__setattr__(
            self,
            '_client_to_sub_trigger',
            MappingProxyType({
                int(client_id): int(index)
                for client_id, index in assignments.items()
            }),
        )
        object.__setattr__(self, 'attacker_client_ids', attackers)

    @property
    def client_to_sub_trigger(self) -> Mapping[int, int]:
        return self._client_to_sub_trigger

    def trigger_for(self, client_id: int) -> ImageTrigger:
        _validate_client_ids((client_id,))
        normalized = int(client_id)
        try:
            index = self.client_to_sub_trigger[normalized]
        except KeyError:
            raise KeyError(f'client {normalized} is not assigned a sub-trigger')
        return self.sub_triggers[index]

    @property
    def full_trigger(self) -> CompositeTrigger:
        return CompositeTrigger(self.sub_triggers)

    def __reduce__(self):
        return (
            type(self),
            (
                self.sub_triggers,
                dict(self.client_to_sub_trigger),
                self.attacker_client_ids,
            ),
        )


class DistributedBackdoorUpdateGenerator:
    """Mark plan-matched poisoned local training as a distributed backdoor update."""

    capabilities = LOCAL_MODEL_CAPABILITIES

    def __init__(
        self,
        *,
        trainer: ScopedLocalTrainer,
        plan: DistributedTriggerPlan,
        source_class: int,
        target_class: int,
        poison_fraction: float,
    ) -> None:
        if not isinstance(trainer, ScopedLocalTrainer):
            raise TypeError('DBA requires ScopedLocalTrainer')
        if not isinstance(plan, DistributedTriggerPlan):
            raise TypeError('plan must be DistributedTriggerPlan')

        normalized_source = class_id(source_class, name='source_class')
        normalized_target = class_id(target_class, name='target_class')
        if normalized_source == normalized_target:
            raise ValueError('source_class and target_class must differ')
        if isinstance(poison_fraction, bool) or not isinstance(poison_fraction, Real):
            raise TypeError('poison_fraction must be a real non-bool scalar')
        normalized_fraction = float(poison_fraction)
        if not math.isfinite(normalized_fraction) or not 0.0 <= normalized_fraction <= 1.0:
            raise ValueError('poison_fraction must be finite and in [0, 1]')

        allowed_client_ids = frozenset(trainer.allowed_client_ids)
        datasets = _validate_training_binding(
            trainer=trainer,
            plan=plan,
            source_class=normalized_source,
            target_class=normalized_target,
            poison_fraction=normalized_fraction,
            client_ids=allowed_client_ids,
        )

        self.trainer = trainer
        self.plan = plan
        self.source_class = normalized_source
        self.target_class = normalized_target
        self.poison_fraction = normalized_fraction
        self._datasets = datasets
        self._clean_datasets = {
            client_id: dataset.dataset for client_id, dataset in datasets.items()
        }
        self._poison_view_states = {
            client_id: (frozenset(dataset.poisoned_indices), int(dataset.eligible_count))
            for client_id, dataset in datasets.items()
        }

    @property
    def allowed_client_ids(self) -> frozenset[int]:
        return self.trainer.allowed_client_ids

    def craft(self, context: AttackContext, rng: RandomSource) -> ClientUpdate:
        datasets = _validate_training_binding(
            trainer=self.trainer,
            plan=self.plan,
            source_class=self.source_class,
            target_class=self.target_class,
            poison_fraction=self.poison_fraction,
            client_ids=(context.client_id,),
            expected_poison_view_states=self._poison_view_states,
        )
        dataset = datasets[context.client_id]
        if dataset is not self._datasets.get(context.client_id):
            raise ValueError(f'client {context.client_id} poisoned dataset replacement')
        if dataset.dataset is not self._clean_datasets.get(context.client_id):
            raise ValueError(f'client {context.client_id} clean dataset replacement')
        base = train_base_update(self.trainer, context, rng)
        metadata = dict(base.metadata)
        metadata.update({
            'attack_type': 'dba',
            'source_class': self.source_class,
            'target_class': self.target_class,
            'poison_fraction': self.poison_fraction,
            'sub_trigger_index': self.plan.client_to_sub_trigger[context.client_id],
            'sub_trigger_count': len(self.plan.sub_triggers),
        })
        return as_malicious_update(base, metadata=metadata)


def _validate_training_binding(
    *,
    trainer: ScopedLocalTrainer,
    plan: DistributedTriggerPlan,
    source_class: int,
    target_class: int,
    poison_fraction: float,
    client_ids: Iterable[int],
    expected_poison_view_states: Mapping[int, tuple[frozenset[int], int]] | None = None,
) -> dict[int, SourceTargetPoisonedDataset]:
    base_trainer = trainer.base_trainer
    if type(base_trainer) is not TorchLocalTrainer:
        raise TypeError('DBA requires exact built-in TorchLocalTrainer')
    if frozenset(trainer.allowed_client_ids) != plan.attacker_client_ids:
        raise ValueError('scoped trainer clients must match distributed trigger plan')

    datasets: dict[int, SourceTargetPoisonedDataset] = {}
    for client_id in client_ids:
        if client_id not in trainer.allowed_client_ids:
            raise ValueError(f'client {client_id} is outside scoped local data')
        dataset = base_trainer.client_datasets.get(client_id)
        if type(dataset) is not SourceTargetPoisonedDataset:
            raise TypeError(
                f'client {client_id} requires exact built-in '
                'SourceTargetPoisonedDataset dataset'
            )
        if (
            dataset.source_class != source_class
            or dataset.target_class != target_class
            or dataset.poison_fraction != poison_fraction
        ):
            raise ValueError(f'client {client_id} poisoned dataset configuration mismatch')
        if dataset.trigger is not plan.trigger_for(client_id):
            raise ValueError(f'client {client_id} poisoned dataset trigger mismatch')
        if expected_poison_view_states is not None:
            current_state = (
                frozenset(dataset.poisoned_indices),
                int(dataset.eligible_count),
            )
            if current_state != expected_poison_view_states.get(client_id):
                raise ValueError(f'client {client_id} poison view state mismatch')
        datasets[client_id] = dataset
    return datasets


def _validate_client_ids(client_ids: Iterable[object]) -> None:
    values = tuple(client_ids)
    if any(
        not isinstance(client_id, Integral) or isinstance(client_id, bool)
        for client_id in values
    ):
        raise TypeError('client ids must be integers')
    if any(int(client_id) < 0 for client_id in values):
        raise ValueError('client ids must be non-negative')


def _validate_indices(indices: Iterable[object], trigger_count: int) -> None:
    values = tuple(indices)
    if any(not isinstance(index, Integral) or isinstance(index, bool) for index in values):
        raise TypeError('sub-trigger indices must be integers')
    if any(int(index) < 0 or int(index) >= trigger_count for index in values):
        raise ValueError('sub-trigger index is out of range')


def _triggers_equal(left: ImageTrigger, right: ImageTrigger) -> bool:
    if left is right:
        return True
    try:
        equality = left == right
        return bool(equality)
    except Exception as error:
        raise TypeError(
            'sub-triggers must support scalar equality for value-distinct validation'
        ) from error
