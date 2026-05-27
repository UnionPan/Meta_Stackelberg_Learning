"""Paper-style distribution learning artifacts for the RL FL attacker.

This module mirrors the distribution-learning stage described in Li et al.,
"Learning to Attack Federated Learning", while keeping it independent from the
paper TD3 deployment path. It reconstructs batches from estimated
aggregate gradients, writes the original reconstructions to ``no_process/``,
denoises them, and writes the denoised images to ``train/`` with ``data.csv``.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.denoiser import (
    ConvDenoisingAutoencoder,
    KerasMnistAutoencoder,
    load_keras_mnist_autoencoder,
    train_denoising_autoencoder,
)
from fl_sandbox.runtime import Weights


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ReconstructorConfig:
    """Configuration for inverting one aggregate-gradient batch."""

    max_iterations: int = 10_000
    lr: float = 0.05
    total_variation: float = 2e-2
    init: str = "zeros"
    optim: str = "adam"
    signed: bool = True
    lr_decay: bool = True
    log_every: int = 0


@dataclass
class ReconstructionResult:
    images: torch.Tensor
    label_logits: torch.Tensor
    loss_history: list[float]

    @property
    def labels(self) -> torch.Tensor:
        return torch.argmax(self.label_logits, dim=1)


@dataclass
class ArtifactResult:
    output_dir: Path
    no_process_dir: Path
    train_dir: Path
    csv_path: Path
    metadata_path: Path
    num_images: int


def estimate_aggregate_gradient(
    previous_weights: Weights,
    current_weights: Weights,
    *,
    lr: float,
    round_gap: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Estimate paper gradient ``(theta_prev - theta_cur) / (eta * gap)``."""

    denominator = max(float(lr) * max(1, int(round_gap)), 1e-12)
    return [
        torch.as_tensor((prev - cur) / denominator, dtype=torch.float32, device=device).detach()
        for prev, cur in zip(previous_weights, current_weights)
    ]


def total_variation(images: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV penalty averaged over the batch."""

    if images.ndim < 4:
        return torch.zeros((), dtype=images.dtype, device=images.device)
    height_tv = torch.mean(torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]))
    width_tv = torch.mean(torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]))
    return height_tv + width_tv


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    values = [tensor.reshape(-1) for tensor in tensors if tensor is not None]
    if not values:
        raise ValueError("cannot flatten an empty tensor sequence")
    return torch.cat(values)


def _soft_cross_entropy(logits: torch.Tensor, label_logits: torch.Tensor) -> torch.Tensor:
    labels = torch.softmax(label_logits, dim=-1)
    return torch.mean(torch.sum(-labels * F.log_softmax(logits, dim=-1), dim=-1))


def _parameter_gradients(loss: torch.Tensor, model: nn.Module) -> tuple[torch.Tensor, ...]:
    params = [param for param in model.parameters() if param.requires_grad]
    gradients = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    return tuple(
        torch.zeros_like(param) if grad is None else grad
        for param, grad in zip(params, gradients)
    )


def _gradient_similarity_loss(
    trial_gradient: Sequence[torch.Tensor],
    input_gradient: Sequence[torch.Tensor],
) -> torch.Tensor:
    count = min(len(trial_gradient), len(input_gradient))
    if count <= 0:
        raise ValueError("input_gradient must contain at least one tensor")
    trial_vec = _flatten_tensors(trial_gradient[:count])
    input_vec = _flatten_tensors(input_gradient[:count]).to(trial_vec.device)
    return 1.0 - F.cosine_similarity(trial_vec, input_vec, dim=0, eps=1e-10)


class PaperGradientReconstructor:
    """Inverting-gradients reconstructor used by the paper-style pipeline."""

    def __init__(
        self,
        *,
        model: nn.Module,
        config: ReconstructorConfig,
        mean: Sequence[float],
        std: Sequence[float],
        num_images: int,
        image_shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.num_images = int(num_images)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = device

    def reconstruct(
        self,
        input_gradient: Sequence[torch.Tensor],
        labels: torch.Tensor | None = None,
        initial_images: torch.Tensor | None = None,
    ) -> ReconstructionResult:
        self.model.eval()
        observed = [tensor.detach().to(self.device) for tensor in input_gradient]
        images = self._initial_images(initial_images)
        images.requires_grad_(True)
        label_logits = self._initial_label_logits(images, labels)
        learn_labels = labels is None
        params: list[torch.Tensor] = [images]
        if learn_labels:
            params.append(label_logits)
        optimizer = self._optimizer(params)
        scheduler = self._scheduler(optimizer)
        loss_history: list[float] = []

        for iteration in range(max(1, int(self.config.max_iterations))):
            optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(images)
            if learn_labels:
                ce_loss = _soft_cross_entropy(logits, label_logits)
            else:
                ce_loss = F.cross_entropy(logits, labels.to(self.device).long())
            gradient = _parameter_gradients(ce_loss, self.model)
            rec_loss = _gradient_similarity_loss(gradient, observed)
            if self.config.total_variation > 0:
                rec_loss = rec_loss + float(self.config.total_variation) * total_variation(images)
            rec_loss.backward()
            if self.config.signed and images.grad is not None:
                images.grad.sign_()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                self._box_images_(images)
            loss_value = float(rec_loss.detach().cpu().item())
            loss_history.append(loss_value)
            if self.config.log_every and iteration % int(self.config.log_every) == 0:
                print(f"reconstruction iteration {iteration}: loss={loss_value:.6f}", flush=True)

        return ReconstructionResult(
            images=images.detach().cpu(),
            label_logits=label_logits.detach().cpu(),
            loss_history=loss_history,
        )

    def _initial_images(self, initial_images: torch.Tensor | None) -> torch.Tensor:
        shape = (self.num_images, *self.image_shape)
        if initial_images is not None:
            images = initial_images.detach().clone().to(self.device, dtype=torch.float32)
            if tuple(images.shape) != shape:
                raise ValueError(f"initial_images shape {tuple(images.shape)} != {shape}")
            return images
        if self.config.init == "zeros":
            images = torch.zeros(shape, dtype=torch.float32, device=self.device)
        elif self.config.init == "rand":
            images = torch.rand(shape, dtype=torch.float32, device=self.device)
        elif self.config.init == "randn":
            images = torch.randn(shape, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"unsupported reconstruction init: {self.config.init}")
        self._box_images_(images)
        return images

    def _initial_label_logits(self, images: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(images)
        if labels is not None:
            labels = labels.to(self.device).long()
            logits = torch.full(
                (self.num_images, output.shape[1]),
                -12.0,
                dtype=torch.float32,
                device=self.device,
            )
            logits.scatter_(1, labels.reshape(-1, 1), 12.0)
            return logits
        return torch.zeros(
            (self.num_images, output.shape[1]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

    def _optimizer(self, params: Sequence[torch.Tensor]) -> torch.optim.Optimizer:
        if self.config.optim.lower() != "adam":
            raise ValueError("paper distribution reconstructor currently supports optim='adam'")
        return torch.optim.Adam(params, lr=float(self.config.lr))

    def _scheduler(self, optimizer: torch.optim.Optimizer):
        if not self.config.lr_decay:
            return None
        max_iterations = max(1, int(self.config.max_iterations))
        milestones = sorted(
            {
                max(1, int(max_iterations // 2.667)),
                max(1, int(max_iterations // 1.6)),
                max(1, int(max_iterations // 1.142)),
            }
        )
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    def _box_images_(self, images: torch.Tensor) -> None:
        mean = torch.as_tensor(self.mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        images.data = torch.max(torch.min(images.data, upper), lower)


def denormalize_images(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.as_tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    return (images.detach().cpu().float() * std_tensor + mean_tensor).clamp(0.0, 1.0)


def normalize_images(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.as_tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    return (images.detach().cpu().float() - mean_tensor) / std_tensor


def write_distribution_artifacts(
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path | str,
    mean: Sequence[float],
    std: Sequence[float],
    metadata: dict[str, object] | None = None,
    denoiser: nn.Module | TensorTransform | None = None,
) -> ArtifactResult:
    """Write paper-style ``no_process`` / ``train`` PNGs and labels."""

    output = Path(output_dir)
    no_process_dir = output / "no_process"
    train_dir = output / "train"
    no_process_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    for directory in (no_process_dir, train_dir):
        for stale_png in directory.glob("*.png"):
            stale_png.unlink()

    labels = labels.detach().cpu().long().reshape(-1)
    if int(images.shape[0]) != int(labels.shape[0]):
        raise ValueError("images and labels must have the same first dimension")
    visible_images = denormalize_images(images, mean, std)
    denoised = _apply_denoiser(visible_images, denoiser)
    for idx in range(int(images.shape[0])):
        _save_image(visible_images[idx], no_process_dir / f"{idx}.png")
        _save_image(denoised[idx], train_dir / f"{idx}.png")

    csv_path = output / "data.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for idx, label in enumerate(labels.tolist()):
            writer.writerow([idx, int(label)])

    payload = {
        "num_images": int(images.shape[0]),
        "no_process_dir": str(no_process_dir),
        "train_dir": str(train_dir),
        "data_csv": str(csv_path),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    if metadata:
        payload.update(metadata)
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return ArtifactResult(
        output_dir=output,
        no_process_dir=no_process_dir,
        train_dir=train_dir,
        csv_path=csv_path,
        metadata_path=metadata_path,
        num_images=int(images.shape[0]),
    )


def _apply_denoiser(images: torch.Tensor, denoiser: nn.Module | TensorTransform | None) -> torch.Tensor:
    if denoiser is None:
        return images.clone()
    with torch.no_grad():
        if isinstance(denoiser, nn.Module):
            denoiser.eval()
            return denoiser(images).detach().cpu().float().clamp(0.0, 1.0)
        return denoiser(images).detach().cpu().float().clamp(0.0, 1.0)


def _save_image(image: torch.Tensor, path: Path) -> None:
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.ndim != 3:
        raise ValueError("expected one image with shape CHW")
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    if array.shape[0] == 1:
        pil_image = Image.fromarray(array[0], mode="L")
    elif array.shape[0] == 3:
        pil_image = Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB")
    else:
        raise ValueError("only 1-channel and 3-channel images are supported")
    pil_image.save(path)


def sample_seed_batch(
    loader,
    *,
    max_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect up to ``max_samples`` clean attacker-local samples."""

    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_images, batch_labels in loader:
        images.append(batch_images.detach().cpu())
        labels.append(batch_labels.detach().cpu().long())
        if sum(int(chunk.shape[0]) for chunk in images) >= int(max_samples):
            break
    if not images:
        raise ValueError("seed loader did not produce any images")
    image_tensor = torch.cat(images, dim=0)[: int(max_samples)].to(device)
    label_tensor = torch.cat(labels, dim=0)[: int(max_samples)].to(device)
    return image_tensor, label_tensor


def shuffled_indices(size: int, count: int, *, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    indices = list(range(int(size)))
    rng.shuffle(indices)
    return indices[: max(0, min(int(count), int(size)))]


def config_as_metadata(config: ReconstructorConfig) -> dict[str, object]:
    return {"reconstructor": asdict(config)}
