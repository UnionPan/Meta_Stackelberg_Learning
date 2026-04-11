"""Shared round-level context objects for attacker-only validation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


Weights = List[np.ndarray]


@dataclass
class RoundContext:
    """Minimum data needed to validate an FL attacker for one round."""

    old_weights: Weights
    benign_weights: List[Weights]
    selected_attacker_ids: List[int]
    model: Optional[Any] = None
    device: Optional[Any] = None
    lr: float = 0.0
    attacker_train_iter: Optional[Any] = None
    poisoned_train_iters: Optional[Dict[str, Any]] = None


@dataclass
class RoundResult:
    """Attack outputs and round-level summary statistics."""

    malicious_weights: List[Weights]
    notes: Optional[str] = None
