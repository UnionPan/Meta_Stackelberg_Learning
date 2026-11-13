"""Design: use the standard ResNet-18 backbone from the Tiny ImageNet experiments.
Data flow: RGB 64x64 -> original 7x7 stride-two stem and maxpool -> 200 logits.
Weights are initialized from scratch; BatchNorm remains enabled.
"""

from torchvision.models import resnet18


def tiny_imagenet_resnet18():
    return resnet18(weights=None, num_classes=200)
