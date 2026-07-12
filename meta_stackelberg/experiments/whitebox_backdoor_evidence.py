"""Held-out clean and source-target metrics for white-box backdoor runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.security.data.labels import class_id
from meta_stackelberg.security.data.trigger import ImageTrigger


@dataclass(frozen=True)
class WhiteBoxBackdoorMetrics:
    clean_loss: float
    clean_accuracy: float
    clean_correct: int
    clean_examples: int
    safe_loss: float
    target_loss: float
    attack_success_rate: float
    attack_successes: int
    source_examples: int
    protocol: str = 'mnist-whitebox-held-out-metrics-v1'

    def __post_init__(self) -> None:
        for name in ('clean_loss', 'safe_loss', 'target_loss'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        for name in ('clean_accuracy', 'attack_success_rate'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f'{name} must be within [0, 1]')
        if self.clean_examples <= 0 or self.source_examples <= 0:
            raise ValueError('held-out example counts must be positive')
        if not 0 <= self.clean_correct <= self.clean_examples:
            raise ValueError('clean_correct is outside clean example count')
        if not 0 <= self.attack_successes <= self.source_examples:
            raise ValueError('attack_successes is outside source example count')


@dataclass(frozen=True)
class WhiteBoxBackdoorMetricEvidence:
    label: str
    query_seeds: tuple[int, ...]
    per_seed: tuple[WhiteBoxBackdoorMetrics, ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError('metric evidence label must not be empty')
        if not self.query_seeds or len(self.query_seeds) != len(self.per_seed):
            raise ValueError('one held-out metric record is required per query seed')
        if len(set(self.query_seeds)) != len(self.query_seeds):
            raise ValueError('query seeds must be unique')

    @property
    def mean_clean_accuracy(self) -> float:
        return sum(item.clean_accuracy for item in self.per_seed) / len(self.per_seed)

    @property
    def mean_attack_success_rate(self) -> float:
        return sum(item.attack_success_rate for item in self.per_seed) / len(self.per_seed)

    @property
    def mean_clean_loss(self) -> float:
        return sum(item.clean_loss for item in self.per_seed) / len(self.per_seed)

    @property
    def mean_safe_loss(self) -> float:
        return sum(item.safe_loss for item in self.per_seed) / len(self.per_seed)

    @property
    def mean_target_loss(self) -> float:
        return sum(item.target_loss for item in self.per_seed) / len(self.per_seed)


@dataclass(frozen=True)
class WhiteBoxSafetyThresholds:
    clean_accuracy_floor: float
    attack_success_rate_ceiling: float
    attack_success_rate_reduction: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f'{name} must be within [0, 1]')


@dataclass(frozen=True)
class WhiteBoxSafetyCheck:
    name: str
    passed: bool
    observed: float
    threshold: float
    comparison: str


@dataclass(frozen=True)
class WhiteBoxSafetyGateResult:
    passed: bool
    checks: tuple[WhiteBoxSafetyCheck, ...]
    thresholds: WhiteBoxSafetyThresholds
    query_seeds: tuple[int, ...]
    protocol: str = 'mnist-whitebox-safety-gate-v1'


def evaluate_whitebox_safety_gate(
    evidence: Mapping[str, WhiteBoxBackdoorMetricEvidence],
    thresholds: WhiteBoxSafetyThresholds,
) -> WhiteBoxSafetyGateResult:
    required = {'learned_defender', 'no_adaptation'}
    missing = required - set(evidence)
    if missing:
        raise ValueError(f'missing white-box metric evidence: {sorted(missing)}')
    learned = evidence['learned_defender']
    baseline = evidence['no_adaptation']
    if learned.label != 'learned_defender' or baseline.label != 'no_adaptation':
        raise ValueError('white-box metric evidence labels do not match keys')
    if learned.query_seeds != baseline.query_seeds:
        raise ValueError('white-box evidence must use matching query seeds')
    reduction = (
        baseline.mean_attack_success_rate - learned.mean_attack_success_rate
    )
    checks = (
        WhiteBoxSafetyCheck(
            'clean_accuracy_floor',
            learned.mean_clean_accuracy >= thresholds.clean_accuracy_floor,
            learned.mean_clean_accuracy,
            thresholds.clean_accuracy_floor,
            'learned clean accuracy >= floor',
        ),
        WhiteBoxSafetyCheck(
            'attack_success_rate_ceiling',
            learned.mean_attack_success_rate
            <= thresholds.attack_success_rate_ceiling,
            learned.mean_attack_success_rate,
            thresholds.attack_success_rate_ceiling,
            'learned ASR <= ceiling',
        ),
        WhiteBoxSafetyCheck(
            'attack_success_rate_reduction',
            reduction >= thresholds.attack_success_rate_reduction,
            reduction,
            thresholds.attack_success_rate_reduction,
            'no-adaptation ASR - learned ASR >= reduction',
        ),
    )
    return WhiteBoxSafetyGateResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        thresholds=thresholds,
        query_seeds=learned.query_seeds,
    )


def evaluate_whitebox_backdoor_model(
    *,
    model: torch.nn.Module,
    query_dataset: Dataset,
    trigger: ImageTrigger,
    source_class: int,
    target_class: int,
    batch_size: int,
) -> WhiteBoxBackdoorMetrics:
    if not isinstance(model, torch.nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if not isinstance(trigger, ImageTrigger):
        raise TypeError('trigger must satisfy ImageTrigger')
    if len(query_dataset) <= 0:
        raise ValueError('query_dataset must not be empty')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    source = class_id(source_class, name='source_class')
    target = class_id(target_class, name='target_class')
    if source == target:
        raise ValueError('source_class and target_class must differ')

    clean_loss_sum = 0.0
    clean_correct = 0
    clean_examples = 0
    safe_loss_sum = 0.0
    target_loss_sum = 0.0
    attack_successes = 0
    source_examples = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for inputs, labels in DataLoader(
                query_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            ):
                labels = labels.long()
                logits = model(inputs)
                clean_loss_sum += float(torch.nn.functional.cross_entropy(
                    logits, labels, reduction='sum',
                ).item())
                predictions = logits.argmax(dim=1)
                clean_correct += int((predictions == labels).sum().item())
                clean_examples += int(labels.numel())

                source_mask = labels == source
                if not bool(source_mask.any()):
                    continue
                triggered = torch.stack([
                    trigger.apply(image) for image in inputs[source_mask]
                ])
                triggered_logits = model(triggered)
                count = int(triggered.shape[0])
                safe_labels = torch.full((count,), source, dtype=torch.long)
                target_labels = torch.full((count,), target, dtype=torch.long)
                safe_loss_sum += float(torch.nn.functional.cross_entropy(
                    triggered_logits, safe_labels, reduction='sum',
                ).item())
                target_loss_sum += float(torch.nn.functional.cross_entropy(
                    triggered_logits, target_labels, reduction='sum',
                ).item())
                attack_successes += int(
                    (triggered_logits.argmax(dim=1) == target).sum().item()
                )
                source_examples += count
    finally:
        model.train(was_training)
    if source_examples <= 0:
        raise ValueError('query_dataset has no source-class examples')
    return WhiteBoxBackdoorMetrics(
        clean_loss=clean_loss_sum / clean_examples,
        clean_accuracy=clean_correct / clean_examples,
        clean_correct=clean_correct,
        clean_examples=clean_examples,
        safe_loss=safe_loss_sum / source_examples,
        target_loss=target_loss_sum / source_examples,
        attack_success_rate=attack_successes / source_examples,
        attack_successes=attack_successes,
        source_examples=source_examples,
    )
