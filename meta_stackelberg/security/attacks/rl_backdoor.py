"""RL-controlled source-to-target malicious local training."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch
from torch.utils.data import Dataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.clients.trainer import TorchLocalTrainer
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.security.attacks.backdoor_action import BackdoorAction
from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
from meta_stackelberg.security.data.trigger import ImageTrigger
from meta_stackelberg.security.types import AttackCapabilities, RoundAttackContext


class RLBackdoorAttack:
    """Apply one decoded BRL action to every sampled malicious client."""

    capabilities = AttackCapabilities(needs_global_model=True, needs_local_data=True)

    def __init__(
        self,
        *,
        action: BackdoorAction,
        model_factory,
        codec: TorchParameterCodec,
        client_datasets: Mapping[int, Dataset],
        trigger: ImageTrigger,
        source_class: int,
        target_class: int,
        batch_size: int,
    ) -> None:
        if not isinstance(action, BackdoorAction):
            raise TypeError('action must be BackdoorAction')
        if not isinstance(codec, TorchParameterCodec):
            raise TypeError('codec must be TorchParameterCodec')
        if not callable(model_factory):
            raise TypeError('model_factory must be callable')
        if not isinstance(trigger, ImageTrigger):
            raise TypeError('trigger must satisfy ImageTrigger')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        datasets = dict(client_datasets)
        if not datasets:
            raise ValueError('client_datasets must not be empty')
        if any(
            isinstance(client_id, bool)
            or not isinstance(client_id, Integral)
            or client_id < 0
            or not isinstance(dataset, Dataset)
            or len(dataset) <= 0
            for client_id, dataset in datasets.items()
        ):
            raise ValueError('client_datasets must contain valid client datasets')
        source = class_id(source_class, name='source_class')
        target = class_id(target_class, name='target_class')
        if source == target:
            raise ValueError('source_class and target_class must differ')
        self.action = action
        self.model_factory = model_factory
        self.codec = codec
        self.client_datasets = {int(key): value for key, value in datasets.items()}
        self.trigger = trigger
        self.source_class = source
        self.target_class = target
        self.batch_size = int(batch_size)

    def craft_round(
        self,
        context: RoundAttackContext,
        rngs: tuple[RandomSource, ...],
    ) -> tuple[ClientUpdate, ...]:
        if context.global_model is None:
            raise ValueError('RL backdoor requires the global model')
        if len(rngs) != len(context.malicious_client_ids):
            raise ValueError('RL backdoor RNG count must match malicious clients')
        missing = set(context.malicious_client_ids) - set(self.client_datasets)
        if missing:
            raise ValueError(f'no local dataset for malicious clients {sorted(missing)}')

        updates = []
        for client_id, rng in zip(context.malicious_client_ids, rngs, strict=True):
            if not isinstance(rng, RandomSource):
                raise TypeError('malicious-client RNGs must be RandomSource instances')
            poisoned = SourceTargetPoisonedDataset(
                dataset=self.client_datasets[client_id],
                trigger=self.trigger,
                source_class=self.source_class,
                target_class=self.target_class,
                poison_fraction=self.action.poison_fraction,
                rng=rng,
            )
            trainer = TorchLocalTrainer(
                model_factory=self.model_factory,
                client_datasets={client_id: poisoned},
                codec=self.codec,
                learning_rate=self.action.learning_rate,
                local_epochs=self.action.local_epochs,
                batch_size=self.batch_size,
            )
            state = RoundState(context.round_index, context.global_model, rng.capture())
            trained = trainer.train(client_id, state, rng)
            metadata = dict(trained.metadata)
            metadata.update({
                'attack_type': 'rl-backdoor',
                'source_class': self.source_class,
                'target_class': self.target_class,
                'poison_fraction': self.action.poison_fraction,
                'malicious_learning_rate': self.action.learning_rate,
                'malicious_local_epochs': self.action.local_epochs,
                'poisoned_count': poisoned.poisoned_count,
                'eligible_source_count': poisoned.eligible_count,
            })
            updates.append(ClientUpdate(
                client_id=client_id,
                delta=trained.delta,
                num_examples=trained.num_examples,
                is_malicious=True,
                metadata=metadata,
            ))
        return tuple(updates)
