"""Disk-backed Phase 1 learned distribution for the paper RL attacker."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _numeric_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


class PaperDistributionDataset:
    """Load completed Phase 1 artifacts from ``root/{split}/*.png`` and ``data.csv``."""

    def __init__(self, root: Path | str, split: str = "train", limit: int | None = None) -> None:
        self.root = Path(root)
        self.split = str(split or "train")
        self.split_dir = self.root / self.split
        if not self.root.is_dir():
            raise ValueError(f"rl_distribution_dir does not exist: {self.root}")
        if not self.split_dir.is_dir():
            raise ValueError(f"Phase 1 distribution split does not exist: {self.split_dir}")

        labels_by_index = self._load_labels()
        mean, std = self._load_normalizer()
        paths = sorted(self.split_dir.glob("*.png"), key=_numeric_sort_key)
        if limit is not None:
            paths = paths[: max(0, int(limit))]
        if not paths:
            raise ValueError(f"No PNG samples found in Phase 1 distribution split: {self.split_dir}")

        images: list[torch.Tensor] = []
        labels: list[int] = []
        for path in paths:
            try:
                sample_id = int(path.stem)
            except ValueError as exc:
                raise ValueError(f"Phase 1 sample filename must be numeric: {path.name}") from exc
            if sample_id not in labels_by_index:
                raise ValueError(f"Missing label for Phase 1 sample id {sample_id} in {self.root / 'data.csv'}")
            image = Image.open(path).convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0)
            tensor = (tensor - mean) / std
            images.append(tensor.to(dtype=torch.float32))
            labels.append(int(labels_by_index[sample_id]))

        self.images = torch.stack(images, dim=0).contiguous()
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def _load_labels(self) -> dict[int, int]:
        label_path = self.root / "data.csv"
        if not label_path.is_file():
            raise ValueError(f"Phase 1 distribution is missing label file: {label_path}")
        labels: dict[int, int] = {}
        with label_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() in {"id", "index", "sample_id"}:
                    continue
                if len(row) < 2:
                    raise ValueError(f"Invalid label row in {label_path}: {row}")
                labels[int(row[0])] = int(row[1])
        return labels

    def _load_normalizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_path = self.root / "metadata.json"
        payload = {}
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        mean = torch.as_tensor(payload.get("mean", [0.0]), dtype=torch.float32).view(-1, 1, 1)
        std = torch.as_tensor(payload.get("std", [1.0]), dtype=torch.float32).view(-1, 1, 1)
        std = torch.clamp(std, min=1e-6)
        if mean.numel() != 1 or std.numel() != 1:
            raise ValueError("PaperDistributionDataset currently supports single-channel MNIST artifacts")
        return mean, std

    def __len__(self) -> int:
        return int(self.labels.numel())

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item() + 1) if len(self) else 0

    @property
    def reconstruction_accept_rate(self) -> float:
        return 1.0

    @property
    def mean_reconstruction_quality(self) -> float:
        return 1.0

    @property
    def accepted_reconstructions(self) -> int:
        return len(self)

    @property
    def rejected_reconstructions(self) -> int:
        return 0

    @property
    def _samples(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (self.images[idx].detach().cpu().clone(), self.labels[idx].detach().cpu().clone())
            for idx in range(len(self))
        ]

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, int(batch_size))
        indices = torch.randint(0, len(self), (count,), dtype=torch.long)
        return self.images[indices].to(device), self.labels[indices].to(device)

    def add_batch(self, *args, **kwargs) -> None:
        return None

    def reject_reconstruction(self, *args, **kwargs) -> None:
        return None


class PaperDistributionSampler:
    """Sampler over the fixed Phase 1 distribution with optional paper growth schedule."""

    def __init__(
        self,
        dataset: PaperDistributionDataset,
        growth_mode: str = "paper_growth",
        *,
        initial_samples: int = 200,
        samples_per_episode: int = 80,
    ) -> None:
        self.dataset = dataset
        self.growth_mode = str(growth_mode or "paper_growth")
        self.initial_samples = max(1, int(initial_samples))
        self.samples_per_episode = max(0, int(samples_per_episode))
        self.episode = 1

    def __len__(self) -> int:
        if self.growth_mode == "full":
            return len(self.dataset)
        if self.growth_mode != "paper_growth":
            raise ValueError(f"Unsupported rl_distribution_growth_mode: {self.growth_mode}")
        limit = self.initial_samples + (max(1, self.episode) - 1) * self.samples_per_episode
        return min(len(self.dataset), max(1, int(limit)))

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def advance_episode(self) -> None:
        self.episode += 1

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        count = max(1, int(batch_size))
        limit = len(self)
        indices = torch.randint(0, limit, (count,), dtype=torch.long)
        return self.dataset.images[indices].to(device), self.dataset.labels[indices].to(device)

