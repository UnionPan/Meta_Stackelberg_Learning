"""Observation construction for the paper-aligned clipped-median RL attacker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _weights_to_vector(weights) -> np.ndarray:
    if weights is None:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate([np.asarray(layer, dtype=np.float32).reshape(-1) for layer in weights]).astype(np.float32)


def build_paper_clipped_median_observation(weights, *, num_attackers: int, tail_layers: int = 2) -> np.ndarray:
    """Original paper state: normalized tail model weights plus selected attacker count."""

    tail = list(weights)[-max(1, int(tail_layers)) :]
    tail_vec = _weights_to_vector(tail)
    state_min = float(np.min(tail_vec)) if tail_vec.size else 0.0
    state_max = float(np.max(tail_vec)) if tail_vec.size else 0.0
    if state_max > state_min:
        norm_state = 2.0 * ((tail_vec - state_min) / (state_max - state_min)) - 1.0
    else:
        norm_state = np.zeros_like(tail_vec, dtype=np.float32)
    return np.concatenate(
        [
            norm_state.astype(np.float32),
            np.asarray([float(num_attackers)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


@dataclass
class FixedRandomProjector:
    """Seeded random projection shared by RL attacker observation builders."""

    output_dim: int
    seed: int
    _matrix: np.ndarray | None = None

    def project(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._matrix is None or self._matrix.shape[1] != vector.shape[0]:
            rng = np.random.default_rng(self.seed)
            scale = 1.0 / np.sqrt(max(1, vector.shape[0]))
            self._matrix = rng.normal(0.0, scale, size=(self.output_dim, vector.shape[0])).astype(np.float32)
        return (self._matrix @ vector).astype(np.float32)
