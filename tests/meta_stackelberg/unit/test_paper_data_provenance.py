import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.experiments.paper_cifar_env import make_generated_cifar_datasets
from meta_stackelberg.experiments.paper_mnist_env import make_generated_mnist_datasets


def _dataset(samples, channels, size):
    return TensorDataset(
        torch.zeros(samples, channels, size, size),
        torch.arange(samples) % 10,
    )


def test_generated_dataset_bundles_record_paper_generator_parameters() -> None:
    mnist = make_generated_mnist_datasets(
        simulated_train=_dataset(20, 1, 28),
        root_dataset=_dataset(5, 1, 28),
        held_out_test=_dataset(10, 1, 28),
        strict_paper_size=False,
    )
    cifar = make_generated_cifar_datasets(
        simulated_train=_dataset(20, 3, 32),
        root_dataset=_dataset(5, 3, 32),
        held_out_test=_dataset(10, 3, 32),
        strict_paper_size=False,
    )
    assert (mnist.provenance.generator, mnist.provenance.generator_training_samples,
            mnist.provenance.generator_epochs) == ('cGAN', 5_000, 100)
    assert (cifar.provenance.generator, cifar.provenance.generator_training_samples,
            cifar.provenance.generator_epochs) == ('conditional-diffusion', 50_000, 30)
    assert mnist.provenance.root_seed_samples == cifar.provenance.root_seed_samples == 200


def test_generated_bundle_strict_mode_rejects_nonpaper_sample_count() -> None:
    with pytest.raises(ValueError, match='60,000'):
        make_generated_mnist_datasets(
            simulated_train=_dataset(20, 1, 28),
            root_dataset=_dataset(5, 1, 28),
            held_out_test=_dataset(10, 1, 28),
        )
