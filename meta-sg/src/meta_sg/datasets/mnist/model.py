"""Design: preserve the existing three-convolution MNIST classifier.
Data flow: [B, 1, 28, 28] -> [B, 64, 14, 14] -> [B, 128, 5, 5] ->
[B, 128, 1, 1] -> [B, 128] -> [B, 10] logits, without softmax.
Convolution and linear layers use Xavier-uniform weights and zero biases.
"""

import torch
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(torch.flatten(features, start_dim=1))
