"""Immutable provenance for paper pre-training and execution datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperDatasetProvenance:
    dataset: str
    source_kind: str
    generator: str
    simulated_train_samples: int
    generator_training_samples: int
    generator_epochs: int
    root_seed_samples: int
    protocol: str = 'paper-dataset-provenance-v1'

    def __post_init__(self) -> None:
        if self.dataset not in {'MNIST', 'CIFAR-10'}:
            raise ValueError('dataset must be MNIST or CIFAR-10')
        if self.source_kind not in {'provided', 'torchvision', 'paper-generated'}:
            raise ValueError('unknown dataset source_kind')
        if not self.generator:
            raise ValueError('generator label must not be empty')
        for name in (
            'simulated_train_samples', 'generator_training_samples',
            'generator_epochs', 'root_seed_samples',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if self.source_kind == 'paper-generated' and (
            self.generator == 'none'
            or self.generator_training_samples <= 0
            or self.generator_epochs <= 0
            or self.root_seed_samples <= 0
        ):
            raise ValueError('paper-generated provenance requires generator training metadata')


def provided_provenance(dataset: str, samples: int) -> PaperDatasetProvenance:
    return PaperDatasetProvenance(
        dataset, 'provided', 'none', samples, 0, 0, 0,
    )
