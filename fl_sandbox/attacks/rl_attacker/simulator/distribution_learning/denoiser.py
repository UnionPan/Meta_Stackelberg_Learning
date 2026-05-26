"""Denoising autoencoders used by the Phase 1 RL attacker simulator.

The paper-style Phase 1 pipeline writes raw gradient-inversion images to
``no_process/`` and denoised images to ``train/``.  This module owns the
denoiser training/loading code so simulator preparation is kept with the
simulator package instead of buried in the distribution writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ConvDenoisingAutoencoder(nn.Module):
    """Small MNIST-style denoising autoencoder."""

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class KerasMnistAutoencoder(nn.Module):
    """PyTorch inference equivalent of the original Keras MNIST autoencoder."""

    layer_names = ("conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4")

    def __init__(self) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2d_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2d_4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv2d(images))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2d_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2d_2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.conv2d_3(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return torch.sigmoid(self.conv2d_4(x))


def build_noisy_images(
    clean_images: torch.Tensor,
    *,
    noise_std: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return clipped clean images plus Gaussian noise in visible ``[0, 1]`` space."""

    clean = clean_images.detach().float().clamp(0.0, 1.0)
    noise = torch.randn(
        clean.shape,
        generator=generator,
        dtype=clean.dtype,
        device=clean.device,
    )
    return (clean + float(noise_std) * noise).clamp(0.0, 1.0)


def train_denoising_autoencoder(
    clean_images: torch.Tensor,
    *,
    noise_std: float = 0.3,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> ConvDenoisingAutoencoder:
    """Train a paper-style denoiser from clean/noisy image pairs."""

    if clean_images.ndim != 4:
        raise ValueError("clean_images must be shaped as NCHW")
    device = device or torch.device("cpu")
    clean = clean_images.detach().float().clamp(0.0, 1.0)
    model = ConvDenoisingAutoencoder(channels=int(clean.shape[1])).to(device)
    loader = DataLoader(TensorDataset(clean), batch_size=max(1, int(batch_size)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    for _ in range(max(0, int(epochs))):
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = build_noisy_images(batch, noise_std=noise_std, generator=generator)
            decoded = model(noisy)
            loss = F.binary_cross_entropy(decoded, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model.cpu()


def save_torch_denoiser(
    model: ConvDenoisingAutoencoder,
    path: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a PyTorch denoiser checkpoint with lightweight metadata."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "fl_sandbox.distribution_learning.denoiser.v1",
            "channels": int(model.channels),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def load_torch_denoiser(
    path: Path | str,
    *,
    device: torch.device | None = None,
) -> tuple[ConvDenoisingAutoencoder, dict[str, Any]]:
    """Load a checkpoint produced by :func:`save_torch_denoiser`."""

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device or torch.device("cpu"))
    model = ConvDenoisingAutoencoder(channels=int(payload.get("channels", 1)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, dict(payload.get("metadata") or {})


def load_keras_mnist_autoencoder(path: Path | str) -> KerasMnistAutoencoder:
    """Load the original Keras ``autoencoder_mnist.h5`` weights without TensorFlow."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - h5py is present in the project env
        raise RuntimeError("h5py is required to load autoencoder_mnist.h5") from exc

    h5_path = Path(path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Keras autoencoder checkpoint not found: {h5_path}")
    model = KerasMnistAutoencoder()
    with h5py.File(h5_path, "r") as handle:
        for layer_name in model.layer_names:
            layer = getattr(model, layer_name)
            group = handle[f"model_weights/{layer_name}/{layer_name}"]
            kernel = torch.as_tensor(group["kernel:0"][()], dtype=layer.weight.dtype)
            bias = torch.as_tensor(group["bias:0"][()], dtype=layer.bias.dtype)
            # Keras Conv2D stores HWIO; PyTorch Conv2d expects OIHW.
            layer.weight.data.copy_(kernel.permute(3, 2, 0, 1).contiguous())
            layer.bias.data.copy_(bias)
    model.eval()
    return model
