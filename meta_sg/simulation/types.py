"""Shared types for FL simulation layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

Weights = List[np.ndarray]


@dataclass
class RoundSummary:
    round_idx: int
    clean_acc: float
    backdoor_acc: float
    clean_loss: float = 0.0
    attack_name: str = "unknown"
    defense_name: str = "unknown"


@dataclass
class SimulationSnapshot:
    round_idx: int
    weights: Weights
    rng_state: Any = None


@dataclass
class InitialState:
    weights: Weights
    round_idx: int = 0


@dataclass(frozen=True)
class SimulationSpec:
    """Static shape metadata for a coordinator without mutating its state."""
    layer_shapes: tuple[tuple[int, ...], ...]

    def empty_weights(self) -> Weights:
        return [np.zeros(shape, dtype=np.float32) for shape in self.layer_shapes]
