"""Robust server-side aggregation rules for sandbox FL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fl_sandbox.core.runtime import Weights

DEFENSE_CHOICES = (
    "fedavg",
    "krum",
    "multi_krum",
    "median",
    "clipped_median",
    "geometric_median",
    "trimmed_mean",
    "fltrust",
)


def weights_to_vector(weights: Weights) -> np.ndarray:
    """Convert a list of NumPy arrays to a 1-D NumPy array."""

    return np.concatenate([np.asarray(weight).ravel() for weight in weights], axis=0)


def vector_to_weights(vector: np.ndarray, weights_template: Weights) -> Weights:
    """Convert a flat vector back into weight tensors following a template."""

    boundaries = np.cumsum([0] + [weight.size for weight in weights_template])
    return [
        np.asarray(vector[boundaries[idx] : boundaries[idx + 1]]).reshape(weights_template[idx].shape)
        for idx in range(len(weights_template))
    ]


def _stack_updates(old_weights: Weights, new_weights: List[Weights]) -> np.ndarray:
    old_vec = weights_to_vector(old_weights)
    return np.stack([old_vec - weights_to_vector(weights) for weights in new_weights], axis=0)


def _clip_updates(updates: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return updates.copy()
    clipped = updates.copy()
    norms = np.linalg.norm(clipped, axis=1)
    for idx, norm in enumerate(norms):
        if norm > max_norm and norm > 0:
            clipped[idx] *= max_norm / norm
    return clipped


def fedavg_aggregate(new_weights: List[Weights]) -> Weights:
    return [
        np.mean([weights[layer_idx] for weights in new_weights], axis=0)
        for layer_idx in range(len(new_weights[0]))
    ]


def median_aggregate(old_weights: Weights, new_weights: List[Weights]) -> Weights:
    updates = _stack_updates(old_weights, new_weights)
    median_update = np.median(updates, axis=0)
    return vector_to_weights(weights_to_vector(old_weights) - median_update, old_weights)


def clipped_median_aggregate(old_weights: Weights, new_weights: List[Weights], max_norm: float) -> Weights:
    updates = _clip_updates(_stack_updates(old_weights, new_weights), max_norm=max_norm)
    median_update = np.median(updates, axis=0)
    return vector_to_weights(weights_to_vector(old_weights) - median_update, old_weights)


def trimmed_mean_aggregate(old_weights: Weights, new_weights: List[Weights], trim_ratio: float) -> Weights:
    updates = _stack_updates(old_weights, new_weights)
    num_updates = updates.shape[0]
    trim = min(int(num_updates * trim_ratio), max(0, (num_updates - 1) // 2))
    if trim == 0:
        mean_update = np.mean(updates, axis=0)
    else:
        sorted_updates = np.sort(updates, axis=0)
        mean_update = np.mean(sorted_updates[trim:num_updates - trim], axis=0)
    return vector_to_weights(weights_to_vector(old_weights) - mean_update, old_weights)


def geometric_median_aggregate(
    old_weights: Weights,
    new_weights: List[Weights],
    max_iters: int = 10,
    eps: float = 1e-6,
) -> Weights:
    vectors = np.stack([weights_to_vector(weights) for weights in new_weights], axis=0)
    current = np.mean(vectors, axis=0)
    for _ in range(max(1, max_iters)):
        distances = np.linalg.norm(vectors - current[None, :], axis=1)
        if np.any(distances < eps):
            current = vectors[np.argmin(distances)]
            break
        coeffs = 1.0 / np.maximum(distances, eps)
        current = np.sum(vectors * coeffs[:, None], axis=0) / np.sum(coeffs)
    return vector_to_weights(current, old_weights)


def _pairwise_sq_dists(updates: np.ndarray) -> np.ndarray:
    num_updates = updates.shape[0]
    dists = np.zeros((num_updates, num_updates), dtype=np.float64)
    for i in range(num_updates):
        diff = updates[i + 1 :] - updates[i]
        sq = np.sum(diff * diff, axis=1)
        dists[i, i + 1 :] = sq
        dists[i + 1 :, i] = sq
    return dists


def _krum_candidate_indices(updates: np.ndarray, num_attackers: int) -> List[int]:
    num_updates = updates.shape[0]
    if num_updates <= 1:
        return [0]

    max_feasible_attackers = max(0, (num_updates - 3) // 2)
    f = min(max(0, int(num_attackers)), max_feasible_attackers)
    neighbor_count = max(1, num_updates - f - 2)
    dists = _pairwise_sq_dists(updates)
    scores = []
    for idx in range(num_updates):
        row = np.sort(np.delete(dists[idx], idx))
        scores.append(np.sum(row[:neighbor_count]))
    return list(np.argsort(np.asarray(scores)))


def krum_aggregate(old_weights: Weights, new_weights: List[Weights], num_attackers: int) -> Weights:
    updates = _stack_updates(old_weights, new_weights)
    best_idx = _krum_candidate_indices(updates, num_attackers=num_attackers)[0]
    chosen_update = updates[best_idx]
    return vector_to_weights(weights_to_vector(old_weights) - chosen_update, old_weights)


def multi_krum_aggregate(
    old_weights: Weights,
    new_weights: List[Weights],
    num_attackers: int,
    num_selected: Optional[int] = None,
) -> Weights:
    updates = _stack_updates(old_weights, new_weights)
    ranked = _krum_candidate_indices(updates, num_attackers=num_attackers)
    max_feasible_attackers = max(0, (updates.shape[0] - 3) // 2)
    f = min(max(0, int(num_attackers)), max_feasible_attackers)
    default_selected = max(1, updates.shape[0] - f - 2)
    selected_count = min(max(1, num_selected or default_selected), len(ranked))
    mean_update = np.mean(updates[ranked[:selected_count]], axis=0)
    return vector_to_weights(weights_to_vector(old_weights) - mean_update, old_weights)


def fltrust_aggregate(old_weights: Weights, new_weights: List[Weights], trusted_weights: Optional[Weights]) -> Weights:
    if trusted_weights is None:
        return fedavg_aggregate(new_weights)

    trusted_update = weights_to_vector(old_weights) - weights_to_vector(trusted_weights)
    trusted_norm = np.linalg.norm(trusted_update)
    if trusted_norm <= 0:
        return fedavg_aggregate(new_weights)

    client_updates = _stack_updates(old_weights, new_weights)
    client_norms = np.linalg.norm(client_updates, axis=1)
    trust_scores = []
    normalized_updates = []
    for update, norm in zip(client_updates, client_norms):
        if norm <= 0:
            trust_scores.append(0.0)
            normalized_updates.append(np.zeros_like(update))
            continue
        cosine = float(np.dot(update, trusted_update) / (norm * trusted_norm))
        score = max(0.0, cosine)
        trust_scores.append(score)
        normalized_updates.append(update * (trusted_norm / norm))

    total_score = float(np.sum(trust_scores))
    if total_score <= 0:
        return fedavg_aggregate(new_weights)

    aggregated_update = np.zeros_like(trusted_update)
    for score, update in zip(trust_scores, normalized_updates):
        aggregated_update += (score / total_score) * update
    return vector_to_weights(weights_to_vector(old_weights) - aggregated_update, old_weights)


@dataclass
class AggregationDefender:
    """Configurable server-side robust aggregator."""

    defense_type: str = "fedavg"
    krum_attackers: int = 1
    multi_krum_selected: Optional[int] = None
    clipped_median_norm: float = 2.0
    trimmed_mean_ratio: float = 0.2
    geometric_median_iters: int = 10

    def aggregate(
        self,
        old_weights: Weights,
        new_weights: List[Weights],
        *,
        trusted_weights: Optional[Weights] = None,
    ) -> Weights:
        if not new_weights:
            return [layer.copy() for layer in old_weights]
        if len(new_weights) == 1:
            return [layer.copy() for layer in new_weights[0]]

        defense_type = self.defense_type.lower()
        if defense_type == "fedavg":
            return fedavg_aggregate(new_weights)
        if defense_type == "krum":
            return krum_aggregate(old_weights, new_weights, num_attackers=self.krum_attackers)
        if defense_type == "multi_krum":
            return multi_krum_aggregate(
                old_weights,
                new_weights,
                num_attackers=self.krum_attackers,
                num_selected=self.multi_krum_selected,
            )
        if defense_type == "median":
            return median_aggregate(old_weights, new_weights)
        if defense_type == "clipped_median":
            return clipped_median_aggregate(old_weights, new_weights, max_norm=self.clipped_median_norm)
        if defense_type == "geometric_median":
            return geometric_median_aggregate(
                old_weights,
                new_weights,
                max_iters=self.geometric_median_iters,
            )
        if defense_type == "trimmed_mean":
            return trimmed_mean_aggregate(
                old_weights,
                new_weights,
                trim_ratio=self.trimmed_mean_ratio,
            )
        if defense_type == "fltrust":
            return fltrust_aggregate(old_weights, new_weights, trusted_weights=trusted_weights)
        raise ValueError(f"Unsupported defense_type: {self.defense_type}")
