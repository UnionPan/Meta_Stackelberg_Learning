"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model selection fields for benchmark-style config composition."""

    dataset: str = "mnist"
    architecture: str = "default"
    num_workers: int = 0


__all__ = ["ModelConfig"]
