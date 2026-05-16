"""Krum-aware projection utilities for the RL attacker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fl_sandbox.core.runtime import Weights
from fl_sandbox.utils.weights import vector_to_weights, weights_to_vector

EPS = 1e-12


@dataclass(frozen=True)
class KrumCandidateStats:
    selected: bool
    rank: int
    score_ratio: float
    feasible_byzantine: int
    neighbor_count: int


@dataclass(frozen=True)
class KrumProjectionResult:
    weights: Weights
    metrics: dict[str, float]


def _typed_vector_weights(vector: np.ndarray, reference: Weights) -> Weights:
    weights = vector_to_weights(vector, reference)
    return [
        np.asarray(layer, dtype=np.asarray(reference[idx]).dtype).copy()
        for idx, layer in enumerate(weights)
    ]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector)
    return vector / norm


def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sum(diff * diff, axis=-1)


def candidate_krum_stats(
    benign_deltas: np.ndarray,
    malicious_delta: np.ndarray,
    *,
    num_attackers: int,
    num_byzantine: int,
    benign_sq_dists: np.ndarray | None = None,
) -> KrumCandidateStats:
    """Krum rank estimate for identical malicious updates."""

    benign_deltas = np.asarray(benign_deltas, dtype=np.float64)
    malicious_delta = np.asarray(malicious_delta, dtype=np.float64).reshape(-1)
    benign_count = int(benign_deltas.shape[0])
    num_attackers = max(0, int(num_attackers))
    if benign_count == 0 or num_attackers <= 0:
        return KrumCandidateStats(False, benign_count, 1.0, 0, 0)

    num_updates = benign_count + num_attackers
    feasible_byzantine = min(max(0, int(num_byzantine)), max(0, (num_updates - 3) // 2))
    neighbor_count = min(max(1, num_updates - feasible_byzantine - 2), num_updates - 1)
    if benign_sq_dists is None:
        benign_sq_dists = _pairwise_sq_dists(benign_deltas)
    benign_sq_dists = np.asarray(benign_sq_dists, dtype=np.float64)

    benign_to_malicious = np.sum((benign_deltas - malicious_delta[None, :]) ** 2, axis=1)
    benign_scores = np.empty(benign_count, dtype=np.float64)
    for idx in range(benign_count):
        benign_neighbors = np.delete(benign_sq_dists[idx], idx)
        row = np.concatenate(
            [
                benign_neighbors,
                np.full(num_attackers, benign_to_malicious[idx], dtype=np.float64),
            ]
        )
        benign_scores[idx] = float(np.sum(np.partition(row, neighbor_count - 1)[:neighbor_count]))

    malicious_row = np.concatenate(
        [
            np.zeros(max(0, num_attackers - 1), dtype=np.float64),
            benign_to_malicious.astype(np.float64, copy=False),
        ]
    )
    malicious_score = float(np.sum(np.partition(malicious_row, neighbor_count - 1)[:neighbor_count]))
    selected = malicious_score < float(np.min(benign_scores))
    rank = int(np.sum(benign_scores < malicious_score))
    median_benign = float(np.median(benign_scores)) if benign_scores.size else EPS
    score_ratio = malicious_score / max(median_benign, EPS)
    if not np.isfinite(score_ratio):
        score_ratio = 1e6
    return KrumCandidateStats(
        bool(selected),
        int(rank),
        float(score_ratio),
        int(feasible_byzantine),
        int(neighbor_count),
    )


def krum_geometry_from_updates(
    update_vectors: np.ndarray,
    malicious_indices: Sequence[int],
    *,
    num_byzantine: int,
) -> KrumCandidateStats:
    update_vectors = np.nan_to_num(np.asarray(update_vectors, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    num_updates = int(update_vectors.shape[0])
    malicious_set = {int(idx) for idx in malicious_indices if 0 <= int(idx) < num_updates}
    if num_updates <= 1 or not malicious_set:
        return KrumCandidateStats(False, max(0, num_updates - 1), 1.0, 0, 0)

    feasible_byzantine = min(max(0, int(num_byzantine)), max(0, (num_updates - 3) // 2))
    neighbor_count = min(max(1, num_updates - feasible_byzantine - 2), num_updates - 1)
    dists = _pairwise_sq_dists(update_vectors)
    scores = np.zeros(num_updates, dtype=np.float64)
    for idx in range(num_updates):
        scores[idx] = float(np.sum(np.sort(np.delete(dists[idx], idx))[:neighbor_count]))

    ranking = np.argsort(scores)
    best_rank = min((rank for rank, idx in enumerate(ranking) if int(idx) in malicious_set), default=num_updates - 1)
    selected = int(ranking[0]) in malicious_set
    benign_scores = [float(scores[idx]) for idx in range(num_updates) if idx not in malicious_set]
    best_benign = min(benign_scores) if benign_scores else float(np.min(scores))
    best_malicious = min(float(scores[idx]) for idx in malicious_set)
    score_ratio = best_malicious / max(best_benign, EPS)
    if not np.isfinite(score_ratio):
        score_ratio = 1e6
    return KrumCandidateStats(
        bool(selected),
        int(best_rank),
        float(score_ratio),
        int(feasible_byzantine),
        int(neighbor_count),
    )


def simulate_krum_benign_surrogates(
    *,
    old_weights: Weights,
    last_aggregate_update: Weights | None,
    benign_count: int,
    seed: int,
    round_idx: int,
    fallback_norm: float = 0.03,
    noise_ratio: float = 0.03,
) -> list[Weights]:
    """Generate deterministic benign update surrogates for fast Krum RL simulation."""

    benign_count = max(0, int(benign_count))
    if benign_count <= 0:
        return []
    old_vec = weights_to_vector(old_weights).astype(np.float64, copy=False)
    if last_aggregate_update is not None:
        anchor_delta = weights_to_vector(last_aggregate_update).astype(np.float64, copy=False)
    else:
        anchor_delta = np.zeros_like(old_vec)
    anchor_delta = np.nan_to_num(anchor_delta, nan=0.0, posinf=0.0, neginf=0.0)
    anchor_norm = float(np.linalg.norm(anchor_delta))
    if anchor_norm <= EPS:
        fallback_seed = int(seed) + 3571 * int(round_idx) + 97 * benign_count
        rng = np.random.default_rng(fallback_seed & 0xFFFF_FFFF)
        anchor_delta = rng.normal(0.0, 1.0, size=old_vec.shape)
        anchor_norm = float(np.linalg.norm(anchor_delta))
        if anchor_norm > EPS:
            anchor_delta = anchor_delta * (float(fallback_norm) / anchor_norm)
            anchor_norm = float(fallback_norm)

    rng_seed = int(seed) + 1009 * int(round_idx) + 37 * benign_count
    rng = np.random.default_rng(rng_seed & 0xFFFF_FFFF)
    synthetic: list[Weights] = []
    for idx in range(benign_count):
        if idx == 0:
            delta = anchor_delta
        else:
            noise = rng.normal(0.0, 1.0, size=anchor_delta.shape)
            noise_norm = float(np.linalg.norm(noise))
            if noise_norm > EPS:
                noise = noise * (float(noise_ratio) * anchor_norm / noise_norm)
            scale = 1.0 + float(rng.normal(0.0, float(noise_ratio)))
            delta = anchor_delta * scale + noise
        synthetic.append(_typed_vector_weights(old_vec + delta, old_weights))
    return synthetic


def _benign_center_and_norm(old_vec: np.ndarray, benign_weights: Sequence[Weights]) -> tuple[np.ndarray, float]:
    if not benign_weights:
        return np.zeros_like(old_vec), 0.0
    benign_deltas = np.stack(
        [weights_to_vector(weights).astype(np.float64, copy=False) - old_vec for weights in benign_weights],
        axis=0,
    )
    benign_deltas = np.nan_to_num(benign_deltas, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(benign_deltas, axis=0), float(np.mean(np.linalg.norm(benign_deltas, axis=1)))


def fast_legacy_krum_raw_surrogate_update(
    *,
    old_weights: Weights,
    benign_weights: Sequence[Weights],
    gamma_scale: float,
    local_steps: int,
    steps_center: float,
    seed: int,
    round_idx: int,
) -> Weights:
    """Cheap approximation of the legacy Krum local-search direction."""

    old_vec = weights_to_vector(old_weights).astype(np.float64, copy=False)
    benign_center, mean_norm = _benign_center_and_norm(old_vec, benign_weights)
    direction = -benign_center
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        rng_seed = int(seed) + 7919 * int(round_idx)
        rng = np.random.default_rng(rng_seed & 0xFFFF_FFFF)
        direction = rng.normal(0.0, 1.0, size=old_vec.shape)
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        raw_delta = np.zeros_like(old_vec)
    else:
        local_step_scale = np.clip(
            float(local_steps) / max(1.0, float(steps_center)),
            0.25,
            2.0,
        )
        scale = float(gamma_scale) * float(local_step_scale) * max(float(mean_norm), EPS)
        raw_delta = scale * direction / direction_norm
    return _typed_vector_weights(old_vec + raw_delta, old_weights)


def fast_legacy_krum_surrogate_update(
    *,
    old_weights: Weights,
    benign_weights: Sequence[Weights],
    gamma_scale: float,
    local_steps: int,
    steps_center: float,
    current_num_attackers: int,
    seed: int,
    round_idx: int,
) -> Weights:
    """Fast simulator-side Krum attack update near the benign cluster boundary."""

    raw = fast_legacy_krum_raw_surrogate_update(
        old_weights=old_weights,
        benign_weights=benign_weights,
        gamma_scale=gamma_scale,
        local_steps=local_steps,
        steps_center=steps_center,
        seed=seed,
        round_idx=round_idx,
    )
    if not benign_weights:
        return raw
    old_vec = weights_to_vector(old_weights).astype(np.float64, copy=False)
    raw_delta = weights_to_vector(raw).astype(np.float64, copy=False) - old_vec
    benign_center, mean_norm = _benign_center_and_norm(old_vec, benign_weights)
    mean_norm = max(float(mean_norm), EPS)
    direction = raw_delta - benign_center
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        direction = -benign_center
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        attack_delta = benign_center
    else:
        attacker_factor = max(1, int(current_num_attackers))
        alpha_cap = 0.25 + 0.35 * attacker_factor
        alpha = min(float(gamma_scale), float(alpha_cap))
        attack_delta = benign_center + alpha * mean_norm * direction / direction_norm
    return _typed_vector_weights(old_vec + attack_delta, old_weights)


def fast_krum_geometry_metrics(*, origin_weights: Weights, weights: Weights) -> tuple[float, float]:
    origin_vec = weights_to_vector(origin_weights).astype(np.float64, copy=False)
    vec = weights_to_vector(weights).astype(np.float64, copy=False)
    loss = float(np.linalg.norm(np.nan_to_num(vec - origin_vec, nan=0.0, posinf=0.0, neginf=0.0)))
    acc = float(np.clip(1.0 - loss, 0.0, 1.0))
    return loss, acc


def project_krum_malicious_update(
    *,
    old_weights: Weights,
    raw_malicious_weights: Weights,
    benign_weights: Sequence[Weights],
    num_attackers: int,
    num_byzantine: int,
    max_alpha: float,
    search_steps: int = 24,
) -> KrumProjectionResult:
    """Project a raw RL update onto a Krum-selectable malicious cluster."""

    if num_attackers <= 0 or not benign_weights:
        return KrumProjectionResult(
            [layer.copy() for layer in raw_malicious_weights],
            {
                "rl_krum_projection_applied": 0.0,
                "rl_krum_projected_selected": 0.0,
                "rl_krum_projected_rank": 0.0,
                "rl_krum_projected_score_ratio": 1.0,
            },
        )

    old_vec = weights_to_vector(old_weights).astype(np.float64, copy=False)
    raw_vec = weights_to_vector(raw_malicious_weights).astype(np.float64, copy=False)
    raw_delta = np.nan_to_num(raw_vec - old_vec, nan=0.0, posinf=0.0, neginf=0.0)
    benign_deltas = np.stack(
        [weights_to_vector(weights).astype(np.float64, copy=False) - old_vec for weights in benign_weights],
        axis=0,
    )
    benign_deltas = np.nan_to_num(benign_deltas, nan=0.0, posinf=0.0, neginf=0.0)
    benign_center = np.mean(benign_deltas, axis=0)
    benign_sq_dists = _pairwise_sq_dists(benign_deltas)
    benign_norms = np.linalg.norm(benign_deltas, axis=1)
    mean_benign_norm = max(float(np.mean(benign_norms)), EPS)

    damage_direction = _normalize(raw_delta - benign_center)
    if not np.any(damage_direction):
        damage_direction = _normalize(raw_delta)
    if not np.any(damage_direction):
        damage_direction = _normalize(-benign_center)

    def candidate_delta(alpha: float) -> np.ndarray:
        return benign_center + float(alpha) * mean_benign_norm * damage_direction

    raw_stats = candidate_krum_stats(
        benign_deltas,
        raw_delta,
        num_attackers=num_attackers,
        num_byzantine=num_byzantine,
        benign_sq_dists=benign_sq_dists,
    )
    max_alpha = max(0.0, float(max_alpha))
    high_stats = candidate_krum_stats(
        benign_deltas,
        candidate_delta(max_alpha),
        num_attackers=num_attackers,
        num_byzantine=num_byzantine,
        benign_sq_dists=benign_sq_dists,
    )
    zero_stats = candidate_krum_stats(
        benign_deltas,
        candidate_delta(0.0),
        num_attackers=num_attackers,
        num_byzantine=num_byzantine,
        benign_sq_dists=benign_sq_dists,
    )

    if high_stats.selected:
        alpha = max_alpha
        projected_stats = high_stats
    elif not zero_stats.selected:
        alpha = 0.0
        projected_stats = zero_stats
    else:
        lo, hi = 0.0, max_alpha
        projected_stats = zero_stats
        for _ in range(max(1, int(search_steps))):
            mid = 0.5 * (lo + hi)
            mid_stats = candidate_krum_stats(
                benign_deltas,
                candidate_delta(mid),
                num_attackers=num_attackers,
                num_byzantine=num_byzantine,
                benign_sq_dists=benign_sq_dists,
            )
            if mid_stats.selected:
                lo = mid
                projected_stats = mid_stats
            else:
                hi = mid
        alpha = lo

    projected_delta = np.nan_to_num(candidate_delta(alpha), nan=0.0, posinf=0.0, neginf=0.0)
    typed_projected = _typed_vector_weights(old_vec + projected_delta, old_weights)
    metrics = {
        "rl_krum_projection_applied": 1.0,
        "rl_krum_raw_selected": float(raw_stats.selected),
        "rl_krum_raw_rank": float(raw_stats.rank),
        "rl_krum_raw_score_ratio": float(raw_stats.score_ratio),
        "rl_krum_projected_selected": float(projected_stats.selected),
        "rl_krum_projected_rank": float(projected_stats.rank),
        "rl_krum_projected_score_ratio": float(projected_stats.score_ratio),
        "rl_krum_projection_alpha": float(alpha),
        "rl_krum_projection_max_alpha": float(max_alpha),
        "rl_krum_raw_delta_norm": float(np.linalg.norm(raw_delta)),
        "rl_krum_projected_delta_norm": float(np.linalg.norm(projected_delta)),
        "rl_krum_mean_benign_norm": float(mean_benign_norm),
        "rl_krum_selected_attackers": float(num_attackers),
        "rl_krum_num_byzantine": float(num_byzantine),
        "rl_krum_feasible_byzantine": float(projected_stats.feasible_byzantine),
        "rl_krum_neighbor_count": float(projected_stats.neighbor_count),
    }
    return KrumProjectionResult(typed_projected, metrics)


__all__ = [
    "KrumCandidateStats",
    "KrumProjectionResult",
    "candidate_krum_stats",
    "fast_krum_geometry_metrics",
    "fast_legacy_krum_raw_surrogate_update",
    "fast_legacy_krum_surrogate_update",
    "krum_geometry_from_updates",
    "project_krum_malicious_update",
    "simulate_krum_benign_surrogates",
]
