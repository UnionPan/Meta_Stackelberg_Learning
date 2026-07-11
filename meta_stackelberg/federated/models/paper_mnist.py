"""MNIST CNN architecture declared in the Meta-SG Appendix C."""

from __future__ import annotations

import torch


class PaperMNISTCNN(torch.nn.Module):
    """8×8, 6×6 and 5×5 convolutional kernels followed by 10 logits."""

    def __init__(self, base_filters: int = 64) -> None:
        super().__init__()
        if base_filters <= 0:
            raise ValueError('base_filters must be positive')
        self.conv1 = torch.nn.Conv2d(
            1, base_filters, kernel_size=8, stride=2, padding=3,
        )
        self.conv2 = torch.nn.Conv2d(
            base_filters, base_filters * 2, kernel_size=6, stride=2,
        )
        self.conv3 = torch.nn.Conv2d(
            base_filters * 2, base_filters * 2, kernel_size=5,
        )
        self.fc1 = torch.nn.Linear(base_filters * 2, 10)
        for module in (self.conv1, self.conv2, self.conv3, self.fc1):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.conv1(inputs))
        values = torch.relu(self.conv2(values))
        values = torch.relu(self.conv3(values))
        return self.fc1(values.flatten(start_dim=1))
