"""Generate paper-style RL attacker distribution-learning artifacts.

The script is intentionally independent from ``RLAttack``.  It runs clean FL
rounds, estimates the aggregate gradient from consecutive global models, applies
an IG-style reconstructor, and writes the visible dataset as:

    output_dir/no_process/*.png
    output_dir/train/*.png
    output_dir/data.csv
    output_dir/metadata.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.config.schema import RunConfig
from fl_sandbox.federation.runner import MinimalFLRunner
from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.core import (
    PaperGradientReconstructor,
    ReconstructorConfig,
    config_as_metadata,
    denormalize_images,
    estimate_aggregate_gradient,
    sample_seed_batch,
    write_distribution_artifacts,
)
from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser import (
    load_keras_mnist_autoencoder,
    train_denoising_autoencoder,
)


DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,), (1, 28, 28)),
    "fmnist": ((0.2860,), (0.3530,), (1, 28, 28)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), (3, 32, 32)),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no_process/train PNGs using paper-style RLFL distribution learning."
    )
    parser.add_argument(
        "--output-dir",
        default="fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1",
        help="Directory where no_process/, train/, data.csv, and metadata.json are written.",
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_STATS), default="mnist")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--noniid-q", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-client-samples-per-client", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--reconstruction-batch-size", type=int, default=32)
    parser.add_argument("--seed-samples", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--inversion-lr", type=float, default=0.05)
    parser.add_argument("--total-variation", type=float, default=2e-2)
    parser.add_argument("--init", choices=("zeros", "rand", "randn", "pre"), default="zeros")
    parser.add_argument(
        "--denoiser-mode",
        choices=("h5", "pytorch", "none"),
        default="h5",
        help="Use original Keras autoencoder H5 weights by default; pytorch is only a fallback mode.",
    )
    parser.add_argument(
        "--autoencoder-path",
        default="fl_sandbox/assets/autoencoder_mnist.h5",
        help="Path to the original Keras autoencoder_mnist.h5 checkpoint.",
    )
    parser.add_argument("--denoiser-epochs", type=int, default=1)
    parser.add_argument("--denoiser-noise-std", type=float, default=0.3)
    parser.add_argument("--denoiser-lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    config.data.dataset = str(args.dataset)
    config.data.split_mode = "paper_q"
    config.data.noniid_q = float(args.noniid_q)
    config.fl.num_clients = int(args.num_clients)
    config.fl.num_attackers = int(args.num_attackers)
    config.fl.subsample_rate = float(args.subsample_rate)
    config.fl.local_epochs = int(args.local_epochs)
    config.runtime.rounds = int(args.rounds)
    config.runtime.device = str(args.device)
    config.runtime.lr = float(args.lr)
    config.runtime.batch_size = int(args.batch_size)
    config.runtime.eval_batch_size = max(1, int(args.max_eval_samples))
    config.runtime.max_client_samples_per_client = (
        None if int(args.max_client_samples_per_client) <= 0 else int(args.max_client_samples_per_client)
    )
    config.runtime.max_eval_samples = None if int(args.max_eval_samples) <= 0 else int(args.max_eval_samples)
    config.runtime.seed = int(args.seed)
    config.attacker.type = "rl"
    config.defender.type = "clipped_median"
    config.defender.clipped_median_norm = 2.0
    return config.normalize()


def sample_pre_initial_images(
    image_pool: torch.Tensor,
    *,
    num_images: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample existing proxy images to initialize the next inversion batch."""

    images, _ = sample_pre_initial_batch(
        image_pool,
        None,
        num_images=num_images,
        device=device,
        generator=generator,
    )
    return images


def sample_pre_initial_batch(
    image_pool: torch.Tensor,
    label_pool: torch.Tensor | None,
    *,
    num_images: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample existing proxy images and their labels using shared indices."""

    if image_pool.ndim != 4:
        raise ValueError("image_pool must be shaped as NCHW")
    if int(image_pool.shape[0]) <= 0:
        raise ValueError("image_pool must contain at least one image")
    labels = None
    if label_pool is not None:
        labels = label_pool.detach().cpu().long().reshape(-1)
        if int(labels.shape[0]) != int(image_pool.shape[0]):
            raise ValueError("image_pool and label_pool must have the same first dimension")
    count = int(num_images)
    if count <= 0:
        raise ValueError("num_images must be positive")
    indices = torch.randint(
        int(image_pool.shape[0]),
        (count,),
        generator=generator,
        device=torch.device("cpu"),
    )
    images = image_pool.detach().cpu().index_select(0, indices).to(device=device, dtype=torch.float32)
    if labels is None:
        return images, None
    return images, labels.index_select(0, indices).to(device=device)


def generate_distribution(args: argparse.Namespace):
    if int(args.seed_samples) <= 0:
        raise ValueError("--seed-samples must be positive so the denoiser has clean attacker data")
    mean, std, image_shape = DATASET_STATS[str(args.dataset)]
    run_config = build_run_config(args)
    recon_config = ReconstructorConfig(
        max_iterations=int(args.max_iterations),
        lr=float(args.inversion_lr),
        total_variation=float(args.total_variation),
        init=str(args.init),
        log_every=int(args.log_every),
    )
    runner = MinimalFLRunner(run_config)
    seed_loader = runner._build_attacker_loader(runner.attacker_ids)  # mirrors attacker-local seed data
    if seed_loader is None:
        seed_loader = runner.client_loaders[0]
    seed_images, seed_labels = sample_seed_batch(
        seed_loader,
        max_samples=int(args.seed_samples),
        device=runner.device,
    )
    visible_seed = denormalize_images(seed_images.cpu(), mean, std)
    denoiser_source = "none"
    denoiser = None
    if args.denoiser_mode == "h5":
        denoiser = load_keras_mnist_autoencoder(Path(args.autoencoder_path))
        denoiser_source = f"keras_h5:{args.autoencoder_path}"
    elif args.denoiser_mode == "pytorch":
        denoiser = train_denoising_autoencoder(
            visible_seed,
            noise_std=float(args.denoiser_noise_std),
            epochs=int(args.denoiser_epochs),
            batch_size=max(1, min(int(args.seed_samples), int(args.batch_size))),
            lr=float(args.denoiser_lr),
            device=runner.device,
        )
        denoiser_source = "fallback_pytorch"

    all_images = [seed_images.detach().cpu()]
    all_labels = [seed_labels.detach().cpu()]
    previous_weights = [weights.copy() for weights in runner.current_weights]
    previous_round = 0
    round_losses: list[float] = []

    for round_idx in range(1, int(args.rounds) + 1):
        runner.run_round(round_idx, attack=None, evaluate=False)
        current_weights = [weights.copy() for weights in runner.current_weights]
        input_gradient = estimate_aggregate_gradient(
            previous_weights,
            current_weights,
            lr=run_config.runtime.lr,
            round_gap=round_idx - previous_round,
            device=runner.device,
        )
        reconstructor = PaperGradientReconstructor(
            model=runner.model,
            config=recon_config,
            mean=mean,
            std=std,
            num_images=int(args.reconstruction_batch_size),
            image_shape=image_shape,
            device=runner.device,
        )
        initial_images = None
        initial_labels = None
        if str(args.init) == "pre":
            initial_images, initial_labels = sample_pre_initial_batch(
                torch.cat(all_images, dim=0),
                torch.cat(all_labels, dim=0),
                num_images=int(args.reconstruction_batch_size),
                device=runner.device,
            )
        result = reconstructor.reconstruct(input_gradient, labels=initial_labels, initial_images=initial_images)
        all_images.append(result.images)
        all_labels.append(result.labels)
        round_losses.append(float(result.loss_history[-1]))
        previous_weights = current_weights
        previous_round = round_idx
        print(
            f"round {round_idx}: reconstructed {result.images.shape[0]} images "
            f"loss={result.loss_history[-1]:.6f}",
            flush=True,
        )

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metadata = {
        "dataset": str(args.dataset),
        "defense": "clipped_median",
        "split_mode": "paper_q",
        "noniid_q": float(args.noniid_q),
        "rounds": int(args.rounds),
        "num_clients": int(args.num_clients),
        "num_attackers": int(args.num_attackers),
        "subsample_rate": float(args.subsample_rate),
        "seed_samples": int(args.seed_samples),
        "reconstruction_batch_size": int(args.reconstruction_batch_size),
        "reconstruction_losses": round_losses,
        "denoiser_source": denoiser_source,
        **config_as_metadata(recon_config),
    }
    return write_distribution_artifacts(
        images=images,
        labels=labels,
        output_dir=Path(args.output_dir),
        mean=mean,
        std=std,
        metadata=metadata,
        denoiser=denoiser,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = generate_distribution(args)
    print(f"wrote {result.num_images} images")
    print(f"no_process: {result.no_process_dir}")
    print(f"train: {result.train_dir}")
    print(f"labels: {result.csv_path}")
    print(f"metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
