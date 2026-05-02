"""Optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Local client optimizer settings."""

    name: str = "sgd"
    lr: float = 0.05
    momentum: float = 0.0
    weight_decay: float = 0.0


__all__ = ["OptimizerConfig"]
