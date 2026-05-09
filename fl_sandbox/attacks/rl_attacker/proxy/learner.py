"""Proxy-distribution learner for adaptive RL attacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.proxy.buffer import ProxyDatasetBuffer
from fl_sandbox.attacks.rl_attacker.proxy.inversion import reconstruct


class ConvDenoiser(nn.Module):
    """Tiny convolutional denoiser used by proxy reconstruction experiments."""

    def __init__(self, noise_std: float = 0.3) -> None:
        super().__init__()
        self.noise_std = float(noise_std)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            images = images + torch.randn_like(images) * self.noise_std
        return self.net(images)


class GradientDistributionLearner:
    """Learns a proxy data distribution from seed data and server updates."""

    def __init__(self, config: RLAttackerConfig) -> None:
        self.config = config
        self.buffer = ProxyDatasetBuffer(limit=config.proxy_buffer_limit)
        self.initialized = False

    def initialize_from_loader(self, loader, device: torch.device) -> None:
        if loader is None:
            self.initialized = True
            return
        added = 0
        for images, labels in loader:
            self.buffer.add_batch(images.to(device), labels.to(device))
            added += int(images.shape[0])
            if added >= self.config.seed_samples:
                break
        self.initialized = True

    def observe_server_update(
        self,
        *,
        previous_weights,
        current_weights,
        model_template,
        server_lr: float,
        round_gap: int,
    ) -> None:
        del previous_weights, current_weights, server_lr, round_gap
        device = next(model_template.parameters()).device
        samples, quality = reconstruct(
            model_template=model_template,
            observed_gradient=None,
            batch_size=self.config.reconstruction_batch_size,
            num_classes=self.buffer.num_classes,
            device=device,
            steps=self.config.inversion_steps,
            lr=self.config.inversion_lr,
        )
        if quality >= self.config.reconstruction_quality_threshold:
            images, labels = samples
            self.buffer.add_batch(images, labels, reconstructed=True)
        else:
            self.buffer.reject_reconstruction(self.config.reconstruction_batch_size)
