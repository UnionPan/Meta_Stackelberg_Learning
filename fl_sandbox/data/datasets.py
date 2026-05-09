"""Dataset helpers internal to ``fl_sandbox``.

The implementation mirrors the historical ``src.utils.data_loader`` behavior
so benchmark code no longer depends on the outer experiment stack.
"""

from __future__ import annotations

from math import floor
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_datasets(data: str, data_dir: str = "data"):
    """Load the configured torchvision dataset."""
    if data == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ])
        return (
            datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform),
            datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform),
        )

    if data == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return (
            datasets.MNIST(root=data_dir, train=True, download=True, transform=transform),
            datasets.MNIST(root=data_dir, train=False, download=True, transform=transform),
        )

    if data == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        train_dataset.targets = torch.LongTensor(train_dataset.targets)
        test_dataset.targets = torch.LongTensor(test_dataset.targets)
        return train_dataset, test_dataset

    raise ValueError(f"Unsupported dataset: {data}")


def add_pattern_bd(x, dataset: str = "cifar10", pattern_type: str = "square", agent_idx: int = -1):
    """Apply the configured backdoor trigger to one raw image."""
    x = np.asarray(x.squeeze()).copy()
    if dataset == "cifar10" and pattern_type == "plus":
        start_idx = 5
        size = 6
        if agent_idx == -1:
            for d in range(0, 3):
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx][d] = 0
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i][d] = 0
        elif agent_idx % 4 == 0:
            for d in range(0, 3):
                for i in range(start_idx, start_idx + (size // 2) + 1):
                    x[i, start_idx][d] = 0
        elif agent_idx % 4 == 1:
            for d in range(0, 3):
                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                    x[i, start_idx][d] = 0
        elif agent_idx % 4 == 2:
            for d in range(0, 3):
                for i in range(start_idx - size // 2, start_idx + size // 4 + 1):
                    x[start_idx + size // 2, i][d] = 0
        elif agent_idx % 4 == 3:
            for d in range(0, 3):
                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i][d] = 0
    elif dataset == "mnist" and pattern_type == "square":
        coords = [(i, j) for i in range(5, 7) for j in range(6, 11)]
        if agent_idx == -1:
            selected_coords = coords
        else:
            splits = [coords[0:3], coords[3:5], coords[5:7], coords[7:10]]
            selected_coords = splits[agent_idx % len(splits)]
        for i, j in selected_coords:
            x[i, j] = 255
    elif dataset == "fmnist":
        if pattern_type == "square":
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255
        elif pattern_type == "plus":
            start_idx = 5
            size = 2
            if agent_idx == -1:
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            elif agent_idx % 4 == 0:
                for i in range(start_idx, start_idx + (size // 2) + 1):
                    x[i, start_idx] = 255
            elif agent_idx % 4 == 1:
                for i in range(start_idx + (size // 2), start_idx + size + 1):
                    x[i, start_idx] = 255
            elif agent_idx % 4 == 2:
                for i in range(start_idx - size // 2, start_idx + size // 4 + 1):
                    x[start_idx + size // 2, i] = 255
            elif agent_idx % 4 == 3:
                for i in range(start_idx - size // 4, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
    return x


def poison_dataset(
    dataset,
    data: str,
    base_class: int,
    target_class: int,
    poison_frac: float,
    pattern_type: str,
    data_idxs=None,
    poison_all: bool = False,
    agent_idx: int = -1,
) -> None:
    """Poison a fraction of base-class examples in-place."""
    all_idxs = (dataset.targets == base_class).nonzero().flatten().tolist()
    if data_idxs is not None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    poison_frac = 1 if poison_all else poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset.data[idx]
        dataset.data[idx] = torch.tensor(
            add_pattern_bd(clean_img, data, pattern_type=pattern_type, agent_idx=agent_idx)
        )
        dataset.targets[idx] = target_class


class DatasetSplit(Dataset):
    """Dataset view containing the examples assigned to one client."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in self.idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]
