"""Deterministic source-to-target poisoned dataset views."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.trigger import ImageTrigger


class SourceTargetPoisonedDataset(Dataset):
    """Apply a trigger and target relabel to a fixed source-only index subset."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        trigger: ImageTrigger,
        source_class: int,
        target_class: int,
        poison_fraction: float,
        rng: RandomSource,
    ) -> None:
        normalized_source = class_id(source_class, name='source_class')
        normalized_target = class_id(target_class, name='target_class')
        if normalized_source == normalized_target:
            raise ValueError('source_class and target_class must be different')
        if not math.isfinite(float(poison_fraction)) or not 0.0 <= poison_fraction <= 1.0:
            raise ValueError('poison_fraction must be finite and within [0, 1]')
        self.dataset = dataset
        self.trigger = trigger
        self.source_class = normalized_source
        self.target_class = normalized_target
        self.poison_fraction = float(poison_fraction)

        eligible = [
            index
            for index in range(len(dataset))
            if _label_as_int(dataset[index][1]) == self.source_class
        ]
        if not eligible:
            raise ValueError('dataset has no eligible source-class examples')
        selected_count = math.floor(self.poison_fraction * len(eligible))
        if selected_count == 0:
            selected: tuple[int, ...] = ()
        else:
            values = rng.numpy.choice(eligible, size=selected_count, replace=False)
            selected = tuple(int(index) for index in values)
        self.eligible_count = len(eligible)
        self.poisoned_indices = frozenset(selected)

    @property
    def poisoned_count(self) -> int:
        return len(self.poisoned_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if index not in self.poisoned_indices:
            return image, label
        triggered = self.trigger.apply(image)
        if isinstance(label, torch.Tensor):
            poisoned_label = torch.as_tensor(
                self.target_class,
                dtype=label.dtype,
                device=label.device,
            )
        else:
            poisoned_label = self.target_class
        return triggered, poisoned_label


def _label_as_int(label) -> int:
    if isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError('classification labels must be scalar')
        return int(label.item())
    return int(label)
