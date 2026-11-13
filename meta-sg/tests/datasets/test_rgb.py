"""Verify RGB preprocessing, official Tiny labels, trigger values, and model inputs."""

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from meta_sg.datasets import cifar10, tiny_imagenet
from meta_sg.datasets.cifar10.triggers import DBA_PATTERNS
from meta_sg.datasets.tiny_imagenet.data import NORMALIZATION


def test_cifar_split_transforms(monkeypatch, tmp_path):
    class CIFARFixture:
        def __init__(self, root, *, train, download, transform):
            self.train, self.transform = train, transform
            self.targets = [0]
            self.data = np.full((1, 32, 32, 3), 127, dtype=np.uint8)

        def __getitem__(self, index):
            return self.transform(Image.fromarray(self.data[index])), self.targets[index]

    monkeypatch.setattr(cifar10.data.datasets, 'CIFAR10', CIFARFixture)
    train, test = cifar10.load_cifar10(tmp_path, augment=True)
    assert train.train and not test.train
    assert isinstance(train.transform.transforms[0], transforms.RandomCrop)
    assert isinstance(test.transform.transforms[0], transforms.ToTensor)
    image, label = test[0]
    torch.testing.assert_close(image, torch.full((3, 32, 32), (127 / 255 - .5) / .5))
    assert label == 0


@pytest.mark.parametrize('normalization', ['legacy', 'tiny'])
def test_tiny_class_mapping_and_normalization(tiny_root, normalization):
    train, val = tiny_imagenet.load_tiny_imagenet(
        tiny_root, train_per_class=2, val_per_class=1, normalization=normalization,
    )
    assert len(train) == 400 and len(val) == 200
    assert train.classes == tuple(sorted(train.classes))
    assert train.data.shape == (400, 3, 64, 64)
    assert train.data.dtype == torch.uint8
    torch.testing.assert_close(train.targets, torch.arange(200).repeat_interleave(2))
    torch.testing.assert_close(val.targets, torch.arange(200))
    mean, std = NORMALIZATION[normalization]
    for data, index in [(train, 18), (val, 9)]:
        image, label = data[index]
        expected = (data.data[index].float() / 255 - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        assert label == 9
        assert bool(data.data[index].eq(9).all())
        assert 'n00000009' in (data.paths[index] if data is train else data.classes[label])
    assert val.paths[9].endswith('val_190.JPEG')


def test_tiny_seeded_selection_and_training_only_augmentation(tiny_root):
    first, val = tiny_imagenet.load_tiny_imagenet(tiny_root, train_per_class=1, val_per_class=1, augment=True)
    second, _ = tiny_imagenet.load_tiny_imagenet(tiny_root, train_per_class=1, val_per_class=1)
    assert first.paths == second.paths
    assert isinstance(first.transform.transforms[0], transforms.RandomRotation)
    assert isinstance(val.transform.transforms[0], transforms.ToTensor)


def test_cifar_trigger_black_union_and_no_input_mutation():
    images = torch.zeros(2, 3, 32, 32)
    full = cifar10.dba_trigger(images)
    combined = images
    for patch in range(4):
        combined = cifar10.dba_trigger(combined, trigger_index=patch)
    torch.testing.assert_close(full, combined)
    assert int(full[0, 0].ne(0).sum()) == len(set(pixel for group in DBA_PATTERNS for pixel in group))
    assert full.min() == -1 and images.count_nonzero() == 0


@pytest.mark.parametrize('normalization', ['legacy', 'tiny'])
def test_tiny_white_trigger_matches_normalization(normalization):
    images = torch.zeros(2, 3, 64, 64)
    full = tiny_imagenet.dba_trigger(images, normalization=normalization)
    combined = images
    for patch in range(4):
        combined = tiny_imagenet.dba_trigger(combined, trigger_index=patch, normalization=normalization)
    torch.testing.assert_close(full, combined)
    assert int(full[0, 0].ne(0).sum()) == 80
    mean, std = NORMALIZATION[normalization]
    torch.testing.assert_close(full[0, :, 0, 0], (1 - torch.tensor(mean)) / torch.tensor(std))
    assert images.count_nonzero() == 0


@pytest.mark.parametrize('model,side,classes', [(cifar10.CIFAR10CNN, 32, 10), (tiny_imagenet.TinyImageNetCNN, 64, 200)])
def test_rgb_models_produce_logits_without_buffers(model, side, classes):
    network = model()
    assert not list(network.buffers())
    with torch.no_grad():
        assert network(torch.zeros(2, 3, side, side)).shape == (2, classes)
