from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.experiments.whitebox_backdoor_evidence import (
    WhiteBoxBackdoorMetricEvidence,
    WhiteBoxBackdoorMetrics,
    WhiteBoxSafetyThresholds,
    evaluate_whitebox_safety_gate,
    evaluate_whitebox_backdoor_model,
)
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger


class TriggerAwareModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(len(inputs), 10)
        logits[:, 1] = 5.0
        triggered = inputs[:, :, 5:7, 6:11].mean(dim=(1, 2, 3)) > 1.0
        logits[triggered, 1] = 0.0
        logits[triggered, 7] = 8.0
        return logits


def test_whitebox_evaluator_reports_clean_and_source_target_metrics() -> None:
    images = torch.zeros(6, 1, 28, 28)
    labels = torch.tensor([1, 1, 1, 1, 2, 3], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    fixture = mnist_global_trigger()

    metrics = evaluate_whitebox_backdoor_model(
        model=TriggerAwareModel(),
        query_dataset=dataset,
        trigger=fixture.trigger,
        source_class=fixture.source_class,
        target_class=fixture.target_class,
        batch_size=2,
    )

    assert metrics.clean_examples == 6
    assert metrics.source_examples == 4
    assert metrics.clean_correct == 4
    assert metrics.clean_accuracy == 4 / 6
    assert metrics.attack_successes == 4
    assert metrics.attack_success_rate == 1.0
    assert metrics.target_loss < metrics.safe_loss


def test_whitebox_evaluator_does_not_mutate_query_images() -> None:
    images = torch.zeros(4, 1, 28, 28)
    dataset = TensorDataset(images, torch.ones(4, dtype=torch.long))
    before = images.clone()
    fixture = mnist_global_trigger()

    evaluate_whitebox_backdoor_model(
        model=TriggerAwareModel(),
        query_dataset=dataset,
        trigger=fixture.trigger,
        source_class=1,
        target_class=7,
        batch_size=4,
    )

    assert torch.equal(images, before)


def _metric(clean_accuracy: float, asr: float) -> WhiteBoxBackdoorMetrics:
    return WhiteBoxBackdoorMetrics(
        clean_loss=0.2,
        clean_accuracy=clean_accuracy,
        clean_correct=int(clean_accuracy * 100),
        clean_examples=100,
        safe_loss=0.3,
        target_loss=0.4,
        attack_success_rate=asr,
        attack_successes=int(asr * 20),
        source_examples=20,
    )


def test_whitebox_safety_gate_requires_clean_floor_asr_ceiling_and_reduction() -> None:
    evidence = {
        'learned_defender': WhiteBoxBackdoorMetricEvidence(
            'learned_defender', (101, 102), (_metric(0.92, 0.1), _metric(0.90, 0.2)),
        ),
        'no_adaptation': WhiteBoxBackdoorMetricEvidence(
            'no_adaptation', (101, 102), (_metric(0.91, 0.6), _metric(0.89, 0.5)),
        ),
    }

    result = evaluate_whitebox_safety_gate(
        evidence,
        WhiteBoxSafetyThresholds(
            clean_accuracy_floor=0.85,
            attack_success_rate_ceiling=0.25,
            attack_success_rate_reduction=0.3,
        ),
    )

    assert result.passed
    assert {check.name for check in result.checks} == {
        'clean_accuracy_floor',
        'attack_success_rate_ceiling',
        'attack_success_rate_reduction',
    }


def test_whitebox_safety_gate_preserves_failed_thresholds() -> None:
    evidence = {
        'learned_defender': WhiteBoxBackdoorMetricEvidence(
            'learned_defender', (101,), (_metric(0.7, 0.8),),
        ),
        'no_adaptation': WhiteBoxBackdoorMetricEvidence(
            'no_adaptation', (101,), (_metric(0.8, 0.9),),
        ),
    }
    result = evaluate_whitebox_safety_gate(
        evidence,
        WhiteBoxSafetyThresholds(0.85, 0.25, 0.3),
    )

    assert not result.passed
    assert all(not check.passed for check in result.checks)
