import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.evaluation.evaluator import test_model as evaluate_model


class ConstantBinaryClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0], 2), dtype=torch.float32, device=inputs.device)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_classification_loss_is_sample_mean(batch_size: int) -> None:
    inputs = torch.zeros((8, 1), dtype=torch.float32)
    labels = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)

    loss, accuracy = evaluate_model(
        ConstantBinaryClassifier(),
        loader,
        device=torch.device('cpu'),
    )

    assert loss == pytest.approx(math.log(2.0), abs=1e-7)
    assert accuracy == pytest.approx(1.0)
