"""Check evaluation mode and sample-weighted classification metrics."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.federated import evaluate


def test_evaluation_counts_an_incomplete_batch_and_disables_dropout():
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
    model = nn.Sequential(nn.Dropout(0.9), linear).train()
    images = torch.tensor([[2., 0.], [0., 1.], [3., 0.], [0., 4.], [0., 2.]])
    labels = torch.tensor([0, 1, 1, 1, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    expected_loss = float(nn.CrossEntropyLoss()(images, labels))

    metrics = evaluate(model, loader, device='cpu')

    assert metrics['examples_seen'] == 5
    assert metrics['accuracy'] == 3 / 5
    assert abs(metrics['loss'] - expected_loss) < 1e-6
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
