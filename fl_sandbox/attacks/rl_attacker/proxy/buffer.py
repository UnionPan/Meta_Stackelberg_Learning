"""Proxy sample buffer for reconstructed and seed attacker data."""

from __future__ import annotations

from collections import deque

import torch


class ProxyDatasetBuffer:
    """Small recency-aware proxy dataset buffer."""

    def __init__(self, limit: int = 2048, num_classes: int = 10) -> None:
        self.limit = int(limit)
        self.num_classes = int(num_classes)
        self._samples: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=self.limit)
        self.accepted_reconstructions = 0
        self.rejected_reconstructions = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def reconstruction_accept_rate(self) -> float:
        total = self.accepted_reconstructions + self.rejected_reconstructions
        return 0.0 if total == 0 else self.accepted_reconstructions / total

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor, *, reconstructed: bool = False) -> None:
        images = images.detach().cpu()
        labels = labels.detach().cpu().long()
        for image, label in zip(images, labels):
            self._samples.append((image.clone(), label.clone()))
        if reconstructed:
            self.accepted_reconstructions += int(images.shape[0])

    def reject_reconstruction(self, count: int = 1) -> None:
        self.rejected_reconstructions += int(count)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._samples:
            images = torch.rand((batch_size, 1, 28, 28), device=device)
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
            return images, labels
        count = min(max(1, int(batch_size)), len(self._samples))
        weights = torch.linspace(0.5, 1.5, steps=len(self._samples))
        indices = torch.multinomial(weights, num_samples=count, replacement=len(self._samples) < count)
        images, labels = zip(*(self._samples[int(idx)] for idx in indices))
        return torch.stack(images).to(device), torch.stack(labels).to(device)
