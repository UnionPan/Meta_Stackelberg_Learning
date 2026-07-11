import math

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.federated.evaluation.classification import ClassificationEvaluator
from meta_stackelberg.federated.models.parameters import TorchParameterCodec


def _zero_model() -> torch.nn.Module:
    model = torch.nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_evaluator_returns_sample_mean_metrics_independent_of_batch_size(batch_size: int) -> None:
    dataset = TensorDataset(
        torch.zeros((8, 1), dtype=torch.float32),
        torch.zeros(8, dtype=torch.long),
    )
    codec = TorchParameterCodec()
    state = codec.capture(_zero_model())

    metrics = ClassificationEvaluator(
        model_factory=_zero_model,
        dataset=dataset,
        codec=codec,
        batch_size=batch_size,
    ).evaluate(state)

    assert metrics.loss == pytest.approx(math.log(2.0), abs=1e-7)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.num_examples == 8


def test_evaluator_handles_empty_dataset_without_non_finite_values() -> None:
    dataset = TensorDataset(
        torch.zeros((0, 1), dtype=torch.float32),
        torch.zeros(0, dtype=torch.long),
    )
    codec = TorchParameterCodec()
    metrics = ClassificationEvaluator(
        model_factory=_zero_model,
        dataset=dataset,
        codec=codec,
        batch_size=4,
    ).evaluate(codec.capture(_zero_model()))

    assert metrics.loss == 0.0
    assert metrics.accuracy == 0.0
    assert metrics.num_examples == 0


def test_evaluation_does_not_mutate_model_state() -> None:
    dataset = TensorDataset(
        torch.ones((2, 1), dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.long),
    )
    codec = TorchParameterCodec()
    state = codec.capture(_zero_model())
    before = state.vector().copy()

    ClassificationEvaluator(
        model_factory=_zero_model,
        dataset=dataset,
        codec=codec,
        batch_size=1,
    ).evaluate(state)

    np.testing.assert_array_equal(state.vector(), before)
