"""Shared round-level context objects for attacker-only validation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader


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
    local_epochs: int = 1
    attacker_train_iter: Optional[DataLoader] = None
    global_poisoned_train_loader: Optional[DataLoader] = None
    sub_trigger_train_loaders: Optional[List[DataLoader]] = None
    poisoned_train_iters: Optional[Dict[str, Any]] = None
    attacker_action: Optional[np.ndarray] = None


@dataclass
class RoundResult:
    """Attack outputs and round-level summary statistics."""

    malicious_weights: List[Weights]
    notes: Optional[str] = None
