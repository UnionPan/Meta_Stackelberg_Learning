"""Design: preserve the existing three-convolution, BatchNorm-free CIFAR classifier.
Data flow: [B, 3, 32, 32] -> 32-channel and 64-channel pooled features ->
[B, 64, 8, 8] -> 128 hidden units -> 10 logits. Xavier weights, zero biases.
"""

import torch
from torch import nn


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(torch.flatten(self.features(images), start_dim=1))
