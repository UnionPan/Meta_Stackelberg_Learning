"""
Paper defense strategy: norm bounding + coordinate-wise trimmed mean.
Post-training: NeuroClip / Pruning (weight-space approximation for stub).

Paper §Appendix C: action space a_D = (α, β, ε/σ).
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from meta_sg.simulation.types import Weights
from meta_sg.strategies.defenses.base import DefenseStrategy
from meta_sg.strategies.types import DefenseDecision


def _stack_updates(old_weights: Weights, new_weights: List[Weights]) -> np.ndarray:
    old_vec = np.concatenate([w.ravel() for w in old_weights])
    return np.stack([
        old_vec - np.concatenate([w.ravel() for w in nw]) for nw in new_weights
    ], axis=0)


def _vec_to_weights(vec: np.ndarray, template: Weights) -> Weights:
    result, idx = [], 0
    for w in template:
        n = w.size
        result.append(vec[idx: idx + n].reshape(w.shape))
        idx += n
    return result


def clip_by_norm(updates: np.ndarray, alpha: float) -> np.ndarray:
    """Clip each row of updates matrix to L2 norm ≤ alpha."""
    if alpha <= 0:
        return updates.copy()
    norms = np.linalg.norm(updates, axis=1, keepdims=True)
    scale = np.minimum(1.0, alpha / (norms + 1e-8))
    return updates * scale


def trimmed_mean(updates: np.ndarray, beta: float) -> np.ndarray:
    """Coordinate-wise trimmed mean: discard beta fraction from each tail."""
    n = updates.shape[0]
    k = min(int(n * beta), max(0, (n - 1) // 2))
    if k == 0 or n <= 2:
        return np.mean(updates, axis=0)
    sorted_u = np.sort(updates, axis=0)
    return np.mean(sorted_u[k: n - k], axis=0)


class PaperDefenseStrategy(DefenseStrategy):
    """
    Paper §Appendix C defense:
      Stage 1 (in coordinator):
        1. Clip all client updates to L2 norm ≤ α
        2. Coordinate-wise trimmed mean with ratio β

      Stage 2 (game layer — reward only):
        Apply NeuroClip(ε) or Prun(σ) on a weight copy
    """

    def aggregate(
        self,
        old_weights: Weights,
        all_weights: List[Weights],
        decision: DefenseDecision,
        trusted_weights: Optional[Weights] = None,
    ) -> Weights:
        if not all_weights:
            return [w.copy() for w in old_weights]

        updates = _stack_updates(old_weights, all_weights)

        # Stage 1a: norm bounding
        clipped = clip_by_norm(updates, decision.norm_bound_alpha)

        # Stage 1b: coordinate-wise trimmed mean
        agg_update = trimmed_mean(clipped, decision.trimmed_mean_beta)

        old_vec = np.concatenate([w.ravel() for w in old_weights])
        new_vec = old_vec - agg_update
        return _vec_to_weights(new_vec, old_weights)

    def apply_post_training(
        self,
        weights: Weights,
        decision: DefenseDecision,
    ) -> Weights:
        """
        Post-training defense on a weight copy.
        For weights (no model structure), we approximate NeuroClip as value clamping
        and Prun as zeroing out smallest-magnitude weights.
        Full model-aware implementation requires a model reference (connect to fl_sandbox).
        """
        w_copy = [w.copy() for w in weights]

        if decision.neuroclip_epsilon is not None:
            eps = decision.neuroclip_epsilon
            w_copy = [np.clip(w, -eps, eps) for w in w_copy]

        elif decision.prun_mask_rate is not None and decision.prun_mask_rate > 0:
            sigma = decision.prun_mask_rate
            for i, w in enumerate(w_copy):
                flat = w.ravel()
                threshold = np.quantile(np.abs(flat), sigma)
                flat[np.abs(flat) < threshold] = 0.0
                w_copy[i] = flat.reshape(w.shape)

        return w_copy
