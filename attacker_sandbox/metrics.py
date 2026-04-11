"""Metrics for standalone attacker validation."""

from typing import Iterable, List

import numpy as np

from src.utils.fl_utils import weights_to_vector


def mean_update_vector(old_weights, weight_list: List[List[np.ndarray]]) -> np.ndarray:
    """Average client update vector in flattened form."""
    if not weight_list:
        return np.zeros_like(weights_to_vector(old_weights))

    updates = [weights_to_vector(old_weights) - weights_to_vector(w) for w in weight_list]
    return np.mean(np.stack(updates, axis=0), axis=0)


def update_norm(old_weights, new_weights: List[np.ndarray]) -> float:
    """L2 norm of a client update."""
    update = weights_to_vector(old_weights) - weights_to_vector(new_weights)
    return float(np.linalg.norm(update))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    denom = max(np.linalg.norm(vec_a) * np.linalg.norm(vec_b), eps)
    return float(np.dot(vec_a, vec_b) / denom)


def update_cosine_to_benign_mean(
    old_weights,
    candidate_weights: List[np.ndarray],
    benign_weights: List[List[np.ndarray]],
) -> float:
    """Cosine similarity between one update and the benign mean update."""
    benign_mean = mean_update_vector(old_weights, benign_weights)
    candidate_update = weights_to_vector(old_weights) - weights_to_vector(candidate_weights)
    return cosine_similarity(candidate_update, benign_mean)


def summarize_norms(old_weights, weights_list: Iterable[List[np.ndarray]]) -> List[float]:
    """Convenience helper for plotting multiple update norms."""
    return [update_norm(old_weights, weights) for weights in weights_list]
