"""Design: preserve the shared framework's lightweight, BatchNorm-free 200-class CNN.
Data flow: [B, 3, 64, 64] -> three convolutions -> adaptive 4x4 pooling ->
128 hidden units -> 200 logits. This is separate from the ResNet-18 experiments.
"""

import torch
from torch import nn


class TinyImageNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 200))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(torch.flatten(self.features(images), start_dim=1))
