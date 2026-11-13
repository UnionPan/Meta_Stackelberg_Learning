"""Design: read the official Tiny ImageNet zip using the existing class mapping.
Data flow: sorted wnids and split annotations -> seeded per-class selection ->
CPU uint8 images -> optional training augmentation -> normalized RGB samples.
Validation annotations supply labels; the unlabeled test split is not evaluated.
"""

from io import BytesIO
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


NORMALIZATION = {
    'legacy': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'tiny': ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


class TinyImageNet(Dataset):
    def __init__(self, data, targets, paths, classes, transform):
        self.data, self.targets = data, targets
        self.paths, self.classes, self.transform = paths, classes, transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = transforms.functional.to_pil_image(self.data[index])
        return self.transform(image), int(self.targets[index])


def _read_split(archive, split, samples_per_class, seed, classes, transform):
    prefix = 'tiny-imagenet-200/'
    candidates = {label: [] for label in classes}
    if split == 'train':
        for name in archive.namelist():
            parts = name.split('/')
            if len(parts) == 5 and parts[1] == 'train' and parts[3] == 'images' and name.lower().endswith('.jpeg'):
                candidates[parts[2]].append(name)
    else:
        for line in archive.read(prefix + 'val/val_annotations.txt').decode().splitlines():
            filename, label, *_ = line.split()
            candidates[label].append(prefix + 'val/images/' + filename)
    generator = random.Random(seed)
    paths, labels = [], []
    for target, label in enumerate(classes):
        selected = generator.sample(sorted(candidates[label]), samples_per_class)
        paths.extend(selected)
        labels.extend([target] * len(selected))
    data = torch.empty((len(paths), 3, 64, 64), dtype=torch.uint8)
    for index, name in enumerate(paths):
        with Image.open(BytesIO(archive.read(name))) as image:
            pixels = np.array(image.convert('RGB'), copy=True)
        data[index] = torch.from_numpy(pixels).permute(2, 0, 1)
    return TinyImageNet(data, torch.tensor(labels), tuple(paths), classes, transform)


def load_tiny_imagenet(
    data_root: str | Path, *, train_per_class=500, val_per_class=50,
    seed=42, normalization='legacy', augment=False,
):
    mean, std = NORMALIZATION[normalization]
    common = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    augmentation = [transforms.RandomRotation(10), transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip()]
    with ZipFile(Path(data_root) / 'tiny-imagenet-200.zip') as archive:
        classes = tuple(sorted(archive.read('tiny-imagenet-200/wnids.txt').decode().split()))
        train = _read_split(archive, 'train', train_per_class, seed, classes,
                            transforms.Compose((augmentation if augment else []) + common))
        val = _read_split(archive, 'val', val_per_class, seed, classes, transforms.Compose(common))
    return train, val
