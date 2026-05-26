"""Phase 1 simulator construction: proxy distribution and denoiser assets."""

from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser import (
    ConvDenoisingAutoencoder,
    KerasMnistAutoencoder,
    build_noisy_images,
    load_keras_mnist_autoencoder,
    load_torch_denoiser,
    save_torch_denoiser,
    train_denoising_autoencoder,
)
from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core import (
    PaperGradientReconstructor,
    ReconstructorConfig,
    estimate_aggregate_gradient,
    write_distribution_artifacts,
)

__all__ = [
    "ConvDenoisingAutoencoder",
    "KerasMnistAutoencoder",
    "PaperGradientReconstructor",
    "ReconstructorConfig",
    "build_noisy_images",
    "estimate_aggregate_gradient",
    "load_keras_mnist_autoencoder",
    "load_torch_denoiser",
    "save_torch_denoiser",
    "train_denoising_autoencoder",
    "write_distribution_artifacts",
]
