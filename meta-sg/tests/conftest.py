"""Small on-disk image datasets for loader and configuration integration checks."""

from io import BytesIO
import struct
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture
def mnist_root(tmp_path):
    raw = tmp_path / 'MNIST/raw'
    raw.mkdir(parents=True)
    for name, count in [('train', 30), ('t10k', 10)]:
        pixels = torch.arange(count * 28 * 28).remainder(256).to(torch.uint8)
        labels = torch.arange(count).remainder(10).to(torch.uint8)
        (raw / f'{name}-images-idx3-ubyte').write_bytes(
            struct.pack('>IIII', 2051, count, 28, 28) + pixels.numpy().tobytes(),
        )
        (raw / f'{name}-labels-idx1-ubyte').write_bytes(
            struct.pack('>II', 2049, count) + labels.numpy().tobytes(),
        )
    return tmp_path


@pytest.fixture
def tiny_root(tmp_path):
    classes = [f'n{index:08d}' for index in range(200)]
    prefix = 'tiny-imagenet-200/'
    annotations = []
    with ZipFile(tmp_path / 'tiny-imagenet-200.zip', 'w') as archive:
        archive.writestr(prefix + 'wnids.txt', '\n'.join(reversed(classes)))
        for label, name in enumerate(classes):
            buffer = BytesIO()
            Image.fromarray(np.full((64, 64), label, dtype=np.uint8)).save(buffer, format='JPEG')
            for sample in range(2):
                archive.writestr(prefix + f'train/{name}/images/{name}_{sample}.JPEG', buffer.getvalue())
            filename = f'val_{199 - label}.JPEG'
            archive.writestr(prefix + f'val/images/{filename}', buffer.getvalue())
            annotations.append(f'{filename}\t{name}\t0\t0\t64\t64')
        archive.writestr(prefix + 'val/val_annotations.txt', '\n'.join(annotations))
    return tmp_path
