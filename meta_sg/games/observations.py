"""State compression: reduce W_t to last-two-layer observation vector."""
from __future__ import annotations

from typing import List

import numpy as np

from meta_sg.simulation.types import Weights


def compress_weights(weights: Weights, num_tail_layers: int = 2) -> np.ndarray:
    """
    Paper §Appendix C: compress w_g^t to its last two hidden layers.
    Extracts the last `num_tail_layers` weight arrays and flattens to 1D.
    """
    if len(weights) == 0:
        return np.zeros(0, dtype=np.float32)
    tail = weights[-num_tail_layers:] if len(weights) >= num_tail_layers else weights
    return np.concatenate([w.ravel().astype(np.float32) for w in tail])


def obs_dim_for(weights: Weights, num_tail_layers: int = 2) -> int:
    """Return the observation dimension for given model weights."""
    return int(compress_weights(weights, num_tail_layers).shape[0])


def normalise_obs(obs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Layer-wise z-score normalisation to keep obs in reasonable range."""
    std = np.std(obs) + eps
    return (obs - np.mean(obs)) / std
