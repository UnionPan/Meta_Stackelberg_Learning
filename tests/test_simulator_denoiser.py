from pathlib import Path

import torch

from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser import (
    ConvDenoisingAutoencoder,
    build_noisy_images,
    load_torch_denoiser,
    save_torch_denoiser,
    train_denoising_autoencoder,
)


def test_build_noisy_images_clamps_to_visible_range_and_preserves_input():
    clean = torch.full((2, 1, 4, 4), 0.9)
    original = clean.clone()
    generator = torch.Generator().manual_seed(7)

    noisy = build_noisy_images(clean, noise_std=0.5, generator=generator)

    assert torch.equal(clean, original)
    assert noisy.shape == clean.shape
    assert torch.min(noisy).item() >= 0.0
    assert torch.max(noisy).item() <= 1.0
    assert not torch.equal(noisy, clean)


def test_train_save_and_load_torch_denoiser_round_trips(tmp_path: Path):
    torch.manual_seed(11)
    clean = torch.rand((4, 1, 8, 8), dtype=torch.float32)

    model = train_denoising_autoencoder(
        clean,
        noise_std=0.1,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
    )

    checkpoint = tmp_path / "autoencoder_mnist.pt"
    save_torch_denoiser(
        model,
        checkpoint,
        metadata={"dataset": "mnist", "seed_samples": 4},
    )
    loaded, metadata = load_torch_denoiser(checkpoint)

    assert isinstance(model, ConvDenoisingAutoencoder)
    assert isinstance(loaded, ConvDenoisingAutoencoder)
    assert metadata["dataset"] == "mnist"
    assert metadata["seed_samples"] == 4
    assert torch.allclose(loaded(clean), model(clean))
