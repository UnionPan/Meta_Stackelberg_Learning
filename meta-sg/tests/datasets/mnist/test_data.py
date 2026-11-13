"""Check local IDX loading, normalization, and the dataset-to-model batch flow."""

import torch
from torch.utils.data import DataLoader, Subset

from meta_sg.data import partition_iid_indices, split_root_indices
from meta_sg.datasets.mnist import MNISTCNN, load_mnist


def test_mnist_samples_preserve_labels_and_existing_normalization(mnist_root):
    train, test = load_mnist(mnist_root)
    assert len(train) == 30
    assert len(test) == 10
    assert train.train and not test.train
    for dataset in [train, test]:
        image, label = dataset[3]
        expected = dataset.data[3].float().div(255).sub(0.1307).div(0.3081).unsqueeze(0)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        assert image.shape == (1, 28, 28)
        assert image.dtype == torch.float32
        assert isinstance(label, int) and label == int(dataset.targets[3])
        assert dataset.data.dtype == torch.uint8


def test_client_subset_batches_feed_the_mnist_model(mnist_root):
    train, _ = load_mnist(mnist_root)
    root, remaining = split_root_indices(train.targets, samples_per_class=1, seed=42)
    partitions = partition_iid_indices(remaining, num_clients=4, seed=42)
    client = Subset(train, partitions[0].tolist())
    images, labels = next(iter(DataLoader(client, batch_size=4)))

    assert root.numel() == 10
    assert len(client) == 5
    assert images.shape == (4, 1, 28, 28)
    assert images.dtype == torch.float32
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64
    torch.testing.assert_close(labels, train.targets[partitions[0][:4]])
    with torch.no_grad():
        assert MNISTCNN()(images).shape == (4, 10)
