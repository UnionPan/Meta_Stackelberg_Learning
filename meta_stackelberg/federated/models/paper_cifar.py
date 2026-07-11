"""CIFAR-10 ResNet-18 architecture used by Meta-SG experiments."""

from __future__ import annotations

import torch


class PaperCIFARBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.bn1(self.conv1(inputs)))
        values = self.bn2(self.conv2(values)) + self.shortcut(inputs)
        return torch.relu(values)


class PaperCIFARResNet18(torch.nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError('num_classes must exceed one')
        self.in_planes = 64
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = torch.nn.Linear(512, num_classes)
        self._initialize()

    def _make_layer(
        self, planes: int, blocks: int, *, stride: int,
    ) -> torch.nn.Sequential:
        strides = (stride,) + (1,) * (blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(PaperCIFARBasicBlock(
                self.in_planes, planes, current_stride,
            ))
            self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu',
                )
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.bn1(self.conv1(inputs)))
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        values = torch.nn.functional.avg_pool2d(values, 4)
        return self.linear(values.flatten(start_dim=1))
