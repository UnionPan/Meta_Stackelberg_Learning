"""Poisoned data preparation facade."""

from __future__ import annotations

from dataclasses import dataclass

from fl_sandbox.data import DatasetSplit, poison_dataset


@dataclass
class PoisonedDataFactory:
    """Holds the poisoning helpers used by backdoor benchmark protocols."""

    dataset_split_cls: type = DatasetSplit
    poison_fn: object = poison_dataset
