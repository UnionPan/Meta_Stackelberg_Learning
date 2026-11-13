"""Design: preserve the shared framework's CIFAR-10 preprocessing.
Data flow: RGB image -> optional crop/flip -> float [3, 32, 32] tensor ->
channel-wise normalization. Return the official training and test splits.
"""

from pathlib import Path

from torchvision import datasets, transforms


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def load_cifar10(data_root: str | Path, *, download=False, augment=False):
    common = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train = datasets.CIFAR10(
        data_root, train=True, download=download,
        transform=transforms.Compose((augmentation if augment else []) + common),
    )
    test = datasets.CIFAR10(data_root, train=False, download=download,
                           transform=transforms.Compose(common))
    return train, test
