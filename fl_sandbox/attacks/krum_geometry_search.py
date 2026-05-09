"""Krum-specific geometry-search attacker.

This is a non-RL heuristic. It searches per-round scaling coefficients for
malicious updates that are likely to be selected by Krum.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from fl_sandbox.attacks.base import SandboxAttack, Weights, get_model_weights, set_model_weights
from fl_sandbox.utils.weights import vector_to_weights, weights_to_vector

EPS = 1e-12


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector)
    return vector / norm


def _delta(old_weights: Weights, new_weights: Weights) -> np.ndarray:
    return weights_to_vector(new_weights) - weights_to_vector(old_weights)


def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sum(diff * diff, axis=-1)


@dataclass(frozen=True)
class KrumGeometry:
    malicious_selected: float
    best_malicious_rank: int
    malicious_score_ratio: float


def krum_geometry(
    update_vectors: np.ndarray,
    malicious_indices: Sequence[int],
    *,
    num_byzantine: int,
) -> KrumGeometry:
    update_vectors = np.nan_to_num(update_vectors, nan=0.0, posinf=1e6, neginf=-1e6)
    num_updates = int(update_vectors.shape[0])
    if num_updates <= 1 or not malicious_indices:
        return KrumGeometry(0.0, max(0, num_updates - 1), 1.0)

    f = min(max(0, int(num_byzantine)), max(0, (num_updates - 3) // 2))
    neighbor_count = min(max(1, num_updates - f - 2), num_updates - 1)
    dists = _pairwise_sq_dists(update_vectors)
    scores = np.zeros(num_updates, dtype=np.float64)
    for idx in range(num_updates):
        scores[idx] = float(np.sum(np.sort(np.delete(dists[idx], idx))[:neighbor_count]))

    ranking = np.argsort(scores)
    malicious_set = {int(idx) for idx in malicious_indices if 0 <= int(idx) < num_updates}
    if not malicious_set:
        return KrumGeometry(0.0, max(0, num_updates - 1), 1.0)
    best_rank = min((rank for rank, idx in enumerate(ranking) if int(idx) in malicious_set), default=num_updates - 1)
    selected = 1.0 if int(ranking[0]) in malicious_set else 0.0
    benign_scores = [float(scores[idx]) for idx in range(num_updates) if idx not in malicious_set]
    best_benign = min(benign_scores) if benign_scores else float(np.min(scores))
    best_malicious = min(float(scores[idx]) for idx in malicious_set)
    return KrumGeometry(selected, int(best_rank), float(best_malicious / max(best_benign, EPS)))


def _krum_selected(
    benign_deltas: np.ndarray,
    malicious_delta: np.ndarray,
    *,
    num_attackers: int,
    num_byzantine: int,
) -> tuple[bool, int, float]:
    benign_sq_dists = _pairwise_sq_dists(benign_deltas)
    return _candidate_krum_stats(
        benign_deltas,
        benign_sq_dists,
        malicious_delta,
        num_attackers=num_attackers,
        num_byzantine=num_byzantine,
    )


def _candidate_krum_stats(
    benign_deltas: np.ndarray,
    benign_sq_dists: np.ndarray,
    malicious_delta: np.ndarray,
    *,
    num_attackers: int,
    num_byzantine: int,
) -> tuple[bool, int, float]:
    """Fast Krum stats for identical malicious updates.

    This avoids building a full pairwise matrix for every candidate alpha.
    """
    benign_count = int(benign_deltas.shape[0])
    if benign_count == 0 or num_attackers <= 0:
        return False, benign_count, 0.0

    num_updates = benign_count + int(num_attackers)
    max_feasible = max(0, (num_updates - 3) // 2)
    f = min(max(0, int(num_byzantine)), max_feasible)
    neighbor_count = min(max(1, num_updates - f - 2), num_updates - 1)

    benign_to_malicious = np.sum((benign_deltas - malicious_delta[None, :]) ** 2, axis=1)
    benign_scores = np.empty(benign_count, dtype=np.float64)
    for idx in range(benign_count):
        benign_neighbors = np.delete(benign_sq_dists[idx], idx)
        row = np.concatenate(
            [
                benign_neighbors,
                np.full(int(num_attackers), benign_to_malicious[idx], dtype=np.float64),
            ]
        )
        benign_scores[idx] = float(np.sum(np.partition(row, neighbor_count - 1)[:neighbor_count]))

    malicious_row = np.concatenate(
        [
            np.zeros(max(0, int(num_attackers) - 1), dtype=np.float64),
            benign_to_malicious.astype(np.float64, copy=False),
        ]
    )
    malicious_score = float(np.sum(np.partition(malicious_row, neighbor_count - 1)[:neighbor_count]))
    median_benign = float(np.median(benign_scores)) if len(benign_scores) else EPS
    selected = malicious_score < float(np.min(benign_scores))
    rank = int(np.sum(benign_scores < malicious_score))
    score_ratio = malicious_score / max(median_benign, EPS)
    if not np.isfinite(score_ratio):
        score_ratio = 1e6
    return bool(selected), rank, float(score_ratio)


def _benign_stats(benign_deltas: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(benign_deltas, axis=1)
    if len(benign_deltas) >= 2:
        dists = np.sqrt(_pairwise_sq_dists(benign_deltas))
        tri = dists[np.triu_indices(len(benign_deltas), k=1)]
        pair_mean = float(np.mean(tri))
        pair_std = float(np.std(tri))
    else:
        pair_mean = 0.0
        pair_std = 0.0
    return {
        "mean_norm": float(np.mean(norms)) if len(norms) else 0.0,
        "std_norm": float(np.std(norms)) if len(norms) else 0.0,
        "pair_mean": pair_mean,
        "pair_std": pair_std,
    }


def _state_from_context(
    ctx,
    *,
    previous_selected: float,
    previous_score_ratio: float,
    previous_alpha: float,
    max_alpha: float,
) -> np.ndarray:
    old_vec = weights_to_vector(ctx.old_weights)
    benign_deltas = np.stack([weights_to_vector(weights) - old_vec for weights in ctx.benign_weights], axis=0)
    stats = _benign_stats(benign_deltas)
    model_norm = float(np.linalg.norm(old_vec)) / max(1.0, float(np.sqrt(len(old_vec))))
    selected_count = float(len(ctx.selected_attacker_ids))
    total_clients = float(max(1, getattr(getattr(ctx, "fl_config", None), "num_clients", 1)))
    round_idx = float(getattr(ctx, "round_idx", 0))
    state = np.asarray(
        [
            round_idx / 1000.0,
            model_norm / 10.0,
            stats["mean_norm"] / 10.0,
            stats["std_norm"] / 10.0,
            stats["pair_mean"] / 10.0,
            stats["pair_std"] / 10.0,
            selected_count / total_clients,
            previous_selected,
            previous_score_ratio / 5.0,
            previous_alpha / max(max_alpha, EPS),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)


def _fallback_damage_delta(old_weights: Weights, benign_weights: List[Weights]) -> np.ndarray:
    old_vec = weights_to_vector(old_weights)
    if not benign_weights:
        return np.zeros_like(old_vec)
    benign_center = np.mean(np.stack([weights_to_vector(weights) - old_vec for weights in benign_weights]), axis=0)
    return -benign_center


def _loss_ascent_delta(ctx, *, ascent_epochs: int = 1, lr_scale: float = 1.0) -> np.ndarray:
    if int(ascent_epochs) <= 0:
        return _fallback_damage_delta(ctx.old_weights, ctx.benign_weights)
    if ctx.model is None or ctx.device is None or ctx.attacker_train_iter is None:
        return _fallback_damage_delta(ctx.old_weights, ctx.benign_weights)
    model = copy.deepcopy(ctx.model).to(ctx.device)
    if getattr(ctx.device, "type", None) == "cuda":
        model = model.to(memory_format=torch.channels_last)
    set_model_weights(model, ctx.old_weights, ctx.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=max(float(ctx.lr) * float(lr_scale), EPS))
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = getattr(ctx.device, "type", None) == "cuda"
    model.train()
    try:
        for _ in range(max(1, int(ctx.local_epochs) * int(ascent_epochs))):
            for images, labels in ctx.attacker_train_iter:
                images = images.to(ctx.device, non_blocking=use_amp)
                labels = labels.to(ctx.device, non_blocking=use_amp)
                if use_amp:
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                (-loss).backward()
                optimizer.step()
    except Exception:
        return _fallback_damage_delta(ctx.old_weights, ctx.benign_weights)
    delta = weights_to_vector(get_model_weights(model)) - weights_to_vector(ctx.old_weights)
    if not np.all(np.isfinite(delta)):
        return _fallback_damage_delta(ctx.old_weights, ctx.benign_weights)
    return np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)


def _craft_weights(ctx, alpha: float, damage_delta: np.ndarray) -> tuple[Weights, Dict[str, float]]:
    old_vec = weights_to_vector(ctx.old_weights)
    benign_deltas = np.stack([weights_to_vector(weights) - old_vec for weights in ctx.benign_weights], axis=0)
    benign_center = np.mean(benign_deltas, axis=0)
    stats = _benign_stats(benign_deltas)
    damage_direction = _normalize(damage_delta)
    if not np.any(damage_direction):
        damage_direction = _normalize(-benign_center)
    attack_delta = benign_center + float(alpha) * max(stats["mean_norm"], EPS) * damage_direction
    attack_delta = np.nan_to_num(attack_delta, nan=0.0, posinf=0.0, neginf=0.0)
    crafted = vector_to_weights(old_vec + attack_delta, ctx.old_weights)
    metrics = {
        "krum_attack_alpha": float(alpha),
        "krum_attack_mean_benign_norm": float(stats["mean_norm"]),
        "krum_attack_delta_norm": float(np.linalg.norm(attack_delta)),
        "krum_attack_damage_cosine": float(np.dot(_normalize(attack_delta), damage_direction)),
    }
    return [np.asarray(layer, dtype=np.asarray(ctx.old_weights[idx]).dtype).copy() for idx, layer in enumerate(crafted)], metrics


@dataclass
class KrumGeometrySearchConfig:
    max_alpha: float = 5.0
    search_steps: int = 24
    ascent_epochs: int = 0
    lr_scale: float = 1.0


class KrumGeometrySearchAttack(SandboxAttack):
    name = "KrumGeometrySearch"
    attack_type = "krum_geometry_search"

    def __init__(self, config: Optional[KrumGeometrySearchConfig] = None) -> None:
        self.config = config or KrumGeometrySearchConfig()
        self.last_action_metrics: Dict[str, float] = {}

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0 or not ctx.benign_weights:
            return self.fallback_old_weights(ctx)

        old_vec = weights_to_vector(ctx.old_weights)
        benign_deltas = np.stack([weights_to_vector(weights) - old_vec for weights in ctx.benign_weights], axis=0)
        damage_delta = _loss_ascent_delta(ctx, ascent_epochs=self.config.ascent_epochs, lr_scale=self.config.lr_scale)
        damage_direction = _normalize(damage_delta)
        if not np.any(damage_direction):
            damage_direction = _normalize(-np.mean(benign_deltas, axis=0))
        benign_center = np.mean(benign_deltas, axis=0)
        benign_sq_dists = _pairwise_sq_dists(benign_deltas)
        mean_norm = max(float(np.mean(np.linalg.norm(benign_deltas, axis=1))), EPS)
        num_byzantine = int(getattr(getattr(ctx, "fl_config", None), "krum_attackers", num_attackers))

        def candidate_delta(alpha: float) -> np.ndarray:
            return benign_center + float(alpha) * mean_norm * damage_direction

        def selected(alpha: float) -> bool:
            is_selected, _, _ = _krum_selected(
                benign_deltas,
                candidate_delta(alpha),
                num_attackers=num_attackers,
                num_byzantine=num_byzantine,
            )
            return is_selected

        def selected_fast(alpha: float) -> bool:
            is_selected, _, _ = _candidate_krum_stats(
                benign_deltas,
                benign_sq_dists,
                candidate_delta(alpha),
                num_attackers=num_attackers,
                num_byzantine=num_byzantine,
            )
            return is_selected

        if selected_fast(self.config.max_alpha):
            alpha = float(self.config.max_alpha)
        elif not selected_fast(0.0):
            alpha = 0.0
        else:
            lo, hi = 0.0, float(self.config.max_alpha)
            for _ in range(max(1, int(self.config.search_steps))):
                mid = 0.5 * (lo + hi)
                if selected_fast(mid):
                    lo = mid
                else:
                    hi = mid
            alpha = lo

        crafted, metrics = _craft_weights(ctx, alpha, damage_delta)
        is_selected, rank, score_ratio = _candidate_krum_stats(
            benign_deltas,
            benign_sq_dists,
            candidate_delta(alpha),
            num_attackers=num_attackers,
            num_byzantine=num_byzantine,
        )
        self.last_action_metrics = {
            **metrics,
            "krum_geometry_search_predicted_selected": float(is_selected),
            "krum_geometry_search_predicted_rank": float(rank),
            "krum_geometry_search_predicted_score_ratio": float(score_ratio),
        }
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]

    def after_round(
        self,
        *,
        ctx,
        all_weights: List[Weights],
        malicious_indices: Sequence[int],
        aggregated_weights: Weights,
        clean_loss: float,
        clean_acc: float,
        num_byzantine: int,
        clean_loss_before: Optional[float] = None,
        clean_acc_before: Optional[float] = None,
        **_: object,
    ) -> Dict[str, float]:
        if not malicious_indices:
            return {}
        update_vectors = np.stack([_delta(ctx.old_weights, weights) for weights in all_weights], axis=0)
        geometry = krum_geometry(update_vectors, malicious_indices, num_byzantine=num_byzantine)
        loss_delta = (
            float(clean_loss) - float(clean_loss_before)
            if clean_loss_before is not None and np.isfinite(clean_loss_before) and np.isfinite(clean_loss)
            else 0.0
        )
        return {
            "krum_geometry_search_selected": float(geometry.malicious_selected),
            "krum_geometry_search_best_malicious_rank": float(geometry.best_malicious_rank),
            "krum_geometry_search_score_ratio": float(geometry.malicious_score_ratio),
            "krum_geometry_search_clean_loss_delta": float(loss_delta),
            "krum_geometry_search_clean_acc": float(clean_acc),
            **self.last_action_metrics,
        }


__all__ = [
    "KrumGeometrySearchAttack",
    "KrumGeometrySearchConfig",
]
