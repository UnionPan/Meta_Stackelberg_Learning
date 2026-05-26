"""Train the Phase 1 MNIST denoiser used by the RL attacker simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser import (
    save_torch_denoiser,
    train_denoising_autoencoder,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 MNIST denoising autoencoder from clean/noisy image pairs."
    )
    parser.add_argument("--dataset", choices=("mnist",), default="mnist")
    parser.add_argument("--data-root", default="fl_sandbox/data")
    parser.add_argument("--output", default="fl_sandbox/assets/autoencoder_mnist.pt")
    parser.add_argument("--seed-samples", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=180)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_mnist_seed_images(
    *,
    data_root: Path | str,
    seed_samples: int,
) -> torch.Tensor:
    """Load the first MNIST seed images as visible NCHW tensors in ``[0, 1]``."""

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - torchvision is part of the project env
        raise RuntimeError("torchvision is required to train the MNIST denoiser") from exc

    dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    count = min(max(1, int(seed_samples)), len(dataset))
    images = [dataset[index][0] for index in range(count)]
    return torch.stack(images).float().clamp(0.0, 1.0)


def train_from_args(args: argparse.Namespace) -> Path:
    torch.manual_seed(int(args.seed))
    device = resolve_device(str(args.device))
    clean_images = load_mnist_seed_images(
        data_root=Path(args.data_root),
        seed_samples=int(args.seed_samples),
    )
    train_count = min(max(1, int(args.train_samples)), clean_images.shape[0])
    clean_train = clean_images[:train_count]
    denoiser = train_denoising_autoencoder(
        clean_train,
        noise_std=float(args.noise_std),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
    )
    return save_torch_denoiser(
        denoiser,
        Path(args.output),
        metadata={
            "dataset": str(args.dataset),
            "data_root": str(args.data_root),
            "seed_samples": int(args.seed_samples),
            "train_samples": int(train_count),
            "noise_std": float(args.noise_std),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
        },
    )


def main(argv: list[str] | None = None) -> None:
    output = train_from_args(parse_args(argv))
    print(f"saved denoiser checkpoint: {output}")


if __name__ == "__main__":
    main()
