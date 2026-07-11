"""Small buffer-free CNN for deterministic clean-FL evidence tests."""

from __future__ import annotations

import torch


class TinyImageCNN(torch.nn.Module):
    """Classify 1×8×8 inputs without BatchNorm or persistent buffers."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be at least two')
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = torch.nn.Linear(8 * 2 * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(start_dim=1))
