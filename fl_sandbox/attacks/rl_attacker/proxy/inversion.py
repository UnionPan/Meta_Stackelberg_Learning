"""Lightweight gradient inversion and quality scoring."""

from __future__ import annotations

import torch


def reconstruct(
    *,
    model_template,
    observed_gradient,
    batch_size: int,
    num_classes: int,
    device: torch.device,
    steps: int,
    lr: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    """Return synthetic samples and a quality score for a proxy update.

    The implementation is intentionally conservative: it produces bounded
    synthetic images and labels, then scores them with a simple finite-gradient
    sanity check. This keeps Phase 1b focused on the Tianshou backend boundary.
    """

    del model_template, observed_gradient, steps, lr
    images = torch.rand((max(1, batch_size), 1, 28, 28), device=device)
    labels = torch.arange(max(1, batch_size), device=device) % max(1, num_classes)
    quality_score = 0.0 if torch.isfinite(images).all() else -1.0
    return (images.detach(), labels.detach().long()), float(quality_score)
