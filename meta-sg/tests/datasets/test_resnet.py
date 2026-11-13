"""Check the two dataset-specific stems and reference ResNet initialization."""

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from meta_sg.datasets.cifar10 import cifar10_resnet18
from meta_sg.datasets.tiny_imagenet import tiny_imagenet_resnet18


@pytest.mark.parametrize('build,side,classes', [(cifar10_resnet18, 32, 10), (tiny_imagenet_resnet18, 64, 200)])
def test_resnet_shapes_and_batchnorm(build, side, classes):
    model = build().eval()
    assert model.fc.out_features == classes
    assert any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    if classes == 10:
        assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
        assert isinstance(model.maxpool, nn.Identity)
    else:
        assert model.conv1.kernel_size == (7, 7) and model.conv1.stride == (2, 2)
        assert isinstance(model.maxpool, nn.MaxPool2d)
    with torch.no_grad():
        assert model(torch.zeros(2, 3, side, side)).shape == (2, classes)


@pytest.mark.parametrize('build,classes', [(cifar10_resnet18, 10), (tiny_imagenet_resnet18, 200)])
def test_resnet_initialization_matches_existing_recipes(build, classes):
    torch.manual_seed(43)
    reference = resnet18(weights=None, num_classes=classes)
    if classes == 10:
        reference.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(reference.conv1.weight, mode='fan_out', nonlinearity='relu')
        reference.maxpool = nn.Identity()
    torch.manual_seed(43)
    actual = build()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
