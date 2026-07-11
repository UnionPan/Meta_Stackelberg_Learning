"""Held-out source-class attack-success-rate evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.security.data.trigger import ImageTrigger
from meta_stackelberg.security.data.labels import class_id


@dataclass(frozen=True)
class TargetedAttackMetrics:
    successes: int
    source_examples: int
    attack_success_rate: float


class TargetedAttackEvaluator:
    """Apply a trigger to held-out source examples and measure target predictions."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], torch.nn.Module],
        dataset: Dataset,
        codec: TorchParameterCodec,
        trigger: ImageTrigger,
        source_class: int,
        target_class: int,
        batch_size: int,
    ) -> None:
        normalized_source = class_id(source_class, name='source_class')
        normalized_target = class_id(target_class, name='target_class')
        if normalized_source == normalized_target:
            raise ValueError('source_class and target_class must differ')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.model_factory = model_factory
        self.dataset = dataset
        self.codec = codec
        self.trigger = trigger
        self.source_class = normalized_source
        self.target_class = normalized_target
        self.batch_size = int(batch_size)

    def evaluate(self, state: ModelState) -> TargetedAttackMetrics:
        triggered_images = []
        for index in range(len(self.dataset)):
            image, label = self.dataset[index]
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            if int(label_value) == self.source_class:
                triggered_images.append(self.trigger.apply(image))
        if not triggered_images:
            raise ValueError('held-out dataset has no source-class examples')

        inputs = torch.stack(triggered_images)
        loader = DataLoader(
            TensorDataset(inputs),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        model = self.model_factory().to('cpu')
        self.codec.load(model, state)
        successes = 0
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                predictions = model(batch).argmax(dim=1)
                successes += int((predictions == self.target_class).sum().item())
        source_examples = len(triggered_images)
        return TargetedAttackMetrics(
            successes=successes,
            source_examples=source_examples,
            attack_success_rate=successes / source_examples,
        )
