"""Design: retain the existing CIFAR-stem ResNet-18 without pretrained weights.
Data flow: RGB 32x32 -> 3x3 stride-one stem, no maxpool -> residual stages -> 10 logits.
BatchNorm buffers are synchronized separately from defended parameter deltas.
"""

from torch import nn
from torchvision.models import resnet18


def cifar10_resnet18():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    return model
