"""Check classification output shape and gradient propagation on synthetic inputs."""

import torch
from torch import nn

from meta_sg.datasets.mnist import MNISTCNN


def test_mnist_cnn_returns_one_logit_per_class() -> None:
    model = MNISTCNN()
    logits = model(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 10)


def test_mnist_cnn_supports_classification_backpropagation() -> None:
    model = MNISTCNN()
    logits = model(torch.randn(2, 1, 28, 28))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
