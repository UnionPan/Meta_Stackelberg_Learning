"""Design: load MNIST through Torchvision and preserve the existing normalization.
Data flow: raw grayscale image -> float tensor in [0, 1] -> normalization ->
(image [1, 28, 28], integer label). Return the official training and test splits.
"""

from pathlib import Path

from torchvision import datasets, transforms


MEAN = 0.1307
STD = 0.3081


def load_mnist(
    data_root: str | Path,
    *,
    download: bool = False,
) -> tuple[datasets.MNIST, datasets.MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])
    train = datasets.MNIST(data_root, train=True, transform=transform, download=download)
    test = datasets.MNIST(data_root, train=False, transform=transform, download=download)
    return train, test
