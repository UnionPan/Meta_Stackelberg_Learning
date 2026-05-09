"""Geometry-search attacker for coordinate-wise clipped median aggregation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def _clip_rows(updates: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return updates.copy()
    clipped = updates.astype(np.float64, copy=True)
    norms = np.linalg.norm(clipped, axis=1)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, EPS))
    return clipped * scale[:, None]


def _clip_one(update: np.ndarray, max_norm: float) -> tuple[np.ndarray, float, float]:
    norm = float(np.linalg.norm(update))
    if max_norm > 0 and norm > max_norm:
        return update * (max_norm / max(norm, EPS)), norm, 1.0
    return update.copy(), norm, 0.0


def _median_with_identical_attack(
    benign_clipped: np.ndarray,
    malicious_update: np.ndarray,
    *,
    num_attackers: int,
) -> np.ndarray:
    if num_attackers <= 0:
        return np.median(benign_clipped, axis=0)
    malicious = np.repeat(malicious_update[None, :], int(num_attackers), axis=0)
    return np.median(np.vstack([benign_clipped, malicious]), axis=0)


def _median_with_attack_matrix(benign_clipped: np.ndarray, malicious_updates: np.ndarray) -> np.ndarray:
    if malicious_updates.size == 0:
        return np.median(benign_clipped, axis=0)
    return np.median(np.vstack([benign_clipped, malicious_updates]), axis=0)


def clipped_median_shift_stats(
    benign_deltas: np.ndarray,
    malicious_delta: np.ndarray,
    *,
    num_attackers: int,
    max_norm: float,
) -> Dict[str, float]:
    """Evaluate the exact clipped-median shift induced by identical attackers."""
    benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
    malicious_clipped, raw_norm, was_clipped = _clip_one(malicious_delta, max_norm=max_norm)
    clean_median = np.median(benign_clipped, axis=0)
    attacked_median = _median_with_identical_attack(
        benign_clipped,
        malicious_clipped,
        num_attackers=num_attackers,
    )
    damage_direction = _normalize(-clean_median)
    if not np.any(damage_direction):
        damage_direction = _normalize(-np.mean(benign_clipped, axis=0))
    shift = attacked_median - clean_median
    damage_gain = float(np.dot(shift, damage_direction))
    clean_alignment = float(np.dot(clean_median, damage_direction))
    attacked_alignment = float(np.dot(attacked_median, damage_direction))
    return {
        "median_geometry_damage_gain": damage_gain,
        "median_geometry_clean_alignment": clean_alignment,
        "median_geometry_attacked_alignment": attacked_alignment,
        "median_geometry_shift_norm": float(np.linalg.norm(shift)),
        "median_geometry_clean_median_norm": float(np.linalg.norm(clean_median)),
        "median_geometry_attacked_median_norm": float(np.linalg.norm(attacked_median)),
        "median_geometry_raw_norm": raw_norm,
        "median_geometry_clipped_norm": float(np.linalg.norm(malicious_clipped)),
        "median_geometry_was_clipped": was_clipped,
    }


def clipped_median_shift_stats_matrix(
    benign_deltas: np.ndarray,
    malicious_deltas: np.ndarray,
    *,
    max_norm: float,
) -> Dict[str, float]:
    """Evaluate the exact clipped-median shift induced by diverse attackers."""
    benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
    malicious_clipped = _clip_rows(malicious_deltas, max_norm=max_norm)
    clean_median = np.median(benign_clipped, axis=0)
    attacked_median = _median_with_attack_matrix(benign_clipped, malicious_clipped)
    damage_direction = _normalize(-clean_median)
    if not np.any(damage_direction):
        damage_direction = _normalize(-np.mean(benign_clipped, axis=0))
    shift = attacked_median - clean_median
    damage_gain = float(np.dot(shift, damage_direction))
    raw_norms = np.linalg.norm(malicious_deltas, axis=1)
    clipped_norms = np.linalg.norm(malicious_clipped, axis=1)
    return {
        "median_geometry_damage_gain": damage_gain,
        "median_geometry_clean_alignment": float(np.dot(clean_median, damage_direction)),
        "median_geometry_attacked_alignment": float(np.dot(attacked_median, damage_direction)),
        "median_geometry_shift_norm": float(np.linalg.norm(shift)),
        "median_geometry_clean_median_norm": float(np.linalg.norm(clean_median)),
        "median_geometry_attacked_median_norm": float(np.linalg.norm(attacked_median)),
        "median_geometry_raw_norm": float(np.mean(raw_norms)) if len(raw_norms) else 0.0,
        "median_geometry_clipped_norm": float(np.mean(clipped_norms)) if len(clipped_norms) else 0.0,
        "median_geometry_was_clipped": float(np.mean(raw_norms > max_norm)) if len(raw_norms) else 0.0,
    }


def _benign_matrix(ctx) -> tuple[np.ndarray, np.ndarray]:
    old_vec = weights_to_vector(ctx.old_weights)
    benign_deltas = np.stack([weights_to_vector(weights) - old_vec for weights in ctx.benign_weights], axis=0)
    return old_vec, benign_deltas


def _clip_norm_from_context(ctx, benign_deltas: np.ndarray, configured: Optional[float] = None) -> float:
    if configured is not None:
        return float(configured)
    configured_value = float(getattr(getattr(ctx, "fl_config", None), "clipped_median_norm", 0.0))
    if configured_value > 0:
        return configured_value
    norms = np.linalg.norm(benign_deltas, axis=1)
    return float(np.median(norms)) if len(norms) else 1.0


def _craft_clipped_median_delta(
    benign_deltas: np.ndarray,
    *,
    alpha: float,
    max_norm: float,
    coordinate_std_floor: float,
    direction_mix: float,
    norm_ratio: float = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
    benign_center = np.mean(benign_clipped, axis=0)
    damage_direction = _normalize(-benign_center)
    if not np.any(damage_direction):
        damage_direction = _normalize(-np.mean(benign_deltas, axis=0))
    coord_median = np.median(benign_clipped, axis=0)
    coord_std = np.std(benign_clipped, axis=0)
    coord_scale = np.maximum(coord_std, max(float(coordinate_std_floor), EPS))
    mean_norm = float(np.mean(np.linalg.norm(benign_clipped, axis=1)))
    sign_direction = np.sign(damage_direction)
    dense_direction = damage_direction * max(mean_norm, EPS)
    rank_candidate = coord_median + float(alpha) * coord_scale * sign_direction
    candidate = (
        (1.0 - float(direction_mix)) * rank_candidate
        + float(direction_mix) * (coord_median + float(alpha) * dense_direction)
    )
    candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    target_norm = max(0.0, float(norm_ratio)) * max_norm
    candidate_norm = float(np.linalg.norm(candidate))
    if target_norm > EPS and candidate_norm > EPS:
        candidate = candidate * min(1.0, target_norm / candidate_norm)
    candidate_clipped, raw_norm, was_clipped = _clip_one(candidate, max_norm=max_norm)
    return candidate_clipped, {
        "median_attack_alpha": float(alpha),
        "median_attack_direction_mix": float(direction_mix),
        "median_attack_norm_ratio": float(norm_ratio),
        "median_attack_raw_norm": float(raw_norm),
        "median_attack_clipped_norm": float(np.linalg.norm(candidate_clipped)),
        "median_attack_was_clipped": float(was_clipped),
        "median_attack_mean_benign_norm": float(mean_norm),
        "median_attack_clip_norm": float(max_norm),
    }


def _scale_to_target_norm(update: np.ndarray, target_norm: float) -> np.ndarray:
    current_norm = float(np.linalg.norm(update))
    if target_norm <= EPS or current_norm <= EPS:
        return update.copy()
    return update * (target_norm / current_norm)


def _topk_mask(priority: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    dim = int(priority.shape[0])
    if dim <= 0:
        return np.zeros_like(priority, dtype=np.float64), 0
    k = max(1, min(dim, int(round(dim * float(np.clip(fraction, 1.0 / dim, 1.0))))))
    if k >= dim:
        return np.ones_like(priority, dtype=np.float64), dim
    indices = np.argpartition(priority, -k)[-k:]
    mask = np.zeros_like(priority, dtype=np.float64)
    mask[indices] = 1.0
    return mask, k


def _fallback_damage_delta(old_weights: Weights, benign_weights: List[Weights]) -> np.ndarray:
    old_vec = weights_to_vector(old_weights)
    if not benign_weights:
        return np.zeros_like(old_vec)
    benign_center = np.mean(np.stack([weights_to_vector(weights) - old_vec for weights in benign_weights]), axis=0)
    return -benign_center


def _loss_ascent_delta(ctx, *, ascent_epochs: int = 1, lr_scale: float = 1.0, max_batches: int = 4) -> np.ndarray:
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
        for _ in range(max(1, int(ascent_epochs))):
            for batch_idx, (images, labels) in enumerate(ctx.attacker_train_iter):
                if batch_idx >= max(1, int(max_batches)):
                    break
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


def _craft_quantile_median_deltas(
    benign_deltas: np.ndarray,
    *,
    num_attackers: int,
    quantile_margin: float,
    extrapolation: float,
    dense_mix: float,
    norm_ratio: float,
    diversity: float,
    anti_median_mix: float,
    sparse_fraction: float,
    loss_ascent_delta: Optional[np.ndarray],
    loss_ascent_mix: float,
    extreme_rank_mix: float,
    max_norm: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
    coord_median = np.median(benign_clipped, axis=0)
    coord_std = np.std(benign_clipped, axis=0)
    damage_direction = _normalize(-coord_median)
    if not np.any(damage_direction):
        damage_direction = _normalize(-np.mean(benign_clipped, axis=0))
    sign_direction = np.sign(damage_direction)
    mean_norm = float(np.mean(np.linalg.norm(benign_clipped, axis=1)))
    dense_target = coord_median + float(extrapolation) * max(mean_norm, EPS) * damage_direction
    anti_target = damage_direction * max_norm
    extreme_probe_value = max(max_norm, float(np.max(np.abs(benign_clipped))) + EPS) * 10.0
    extreme_probe = (np.where(sign_direction == 0.0, np.sign(-coord_median), sign_direction) * extreme_probe_value)
    extreme_attacked = _median_with_identical_attack(
        benign_clipped,
        extreme_probe,
        num_attackers=num_attackers,
    )
    rank_shift = extreme_attacked - coord_median
    priority = np.maximum(rank_shift * damage_direction, 0.0) + 0.05 * coord_std
    sparse_mask, sparse_k = _topk_mask(priority, fraction=sparse_fraction)
    sparse_sign = np.where(sign_direction == 0.0, np.sign(-coord_median), sign_direction) * sparse_mask
    extreme_target = _scale_to_target_norm(sparse_sign, target_norm=max_norm)
    if loss_ascent_delta is not None and loss_ascent_delta.shape == coord_median.shape:
        sparse_loss_direction = _normalize(loss_ascent_delta * sparse_mask)
    else:
        sparse_loss_direction = np.zeros_like(coord_median)
    loss_target = sparse_loss_direction * max_norm

    margin = float(np.clip(quantile_margin, 0.0, 1.0))
    base_hi = 0.5 + 0.49 * margin
    base_lo = 0.5 - 0.49 * margin
    spread = 0.24 * float(np.clip(diversity, 0.0, 1.0))
    offsets = np.linspace(-spread, spread, max(1, int(num_attackers)))
    target_norm = float(np.clip(norm_ratio, 0.05, 1.2)) * max_norm

    malicious = []
    q_values = []
    raw_norms = []
    clipped_flags = []
    for offset in offsets:
        q_hi = float(np.clip(base_hi + offset, 0.5, 0.995))
        q_lo = float(np.clip(base_lo - offset, 0.005, 0.5))
        high = np.quantile(benign_clipped, q_hi, axis=0)
        low = np.quantile(benign_clipped, q_lo, axis=0)
        quantile_target = np.where(sign_direction >= 0.0, high, low)
        rank_target = coord_median + float(extrapolation) * (quantile_target - coord_median)
        candidate = (1.0 - float(dense_mix)) * rank_target + float(dense_mix) * dense_target
        candidate = (1.0 - float(anti_median_mix)) * candidate + float(anti_median_mix) * anti_target
        candidate = candidate * sparse_mask
        candidate = (1.0 - float(extreme_rank_mix)) * candidate + float(extreme_rank_mix) * extreme_target
        candidate = (1.0 - float(loss_ascent_mix)) * candidate + float(loss_ascent_mix) * loss_target
        candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        candidate = _scale_to_target_norm(candidate, target_norm=target_norm)
        candidate, raw_norm, was_clipped = _clip_one(candidate, max_norm=max_norm)
        malicious.append(candidate)
        q_values.append((q_hi + (1.0 - q_lo)) * 0.5)
        raw_norms.append(raw_norm)
        clipped_flags.append(was_clipped)

    matrix = np.stack(malicious, axis=0) if malicious else np.zeros((0, benign_deltas.shape[1]), dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) if len(matrix) else np.asarray([], dtype=np.float64)
    return matrix, {
        "median_attack_quantile_margin": margin,
        "median_attack_extrapolation": float(extrapolation),
        "median_attack_direction_mix": float(dense_mix),
        "median_attack_norm_ratio": float(norm_ratio),
        "median_attack_diversity": float(diversity),
        "median_attack_anti_median_mix": float(anti_median_mix),
        "median_attack_sparse_fraction": float(sparse_fraction),
        "median_attack_sparse_k": float(sparse_k),
        "median_attack_loss_ascent_mix": float(loss_ascent_mix),
        "median_attack_extreme_rank_mix": float(extreme_rank_mix),
        "median_attack_loss_ascent_norm": float(np.linalg.norm(loss_ascent_delta)) if loss_ascent_delta is not None else 0.0,
        "median_attack_mean_q": float(np.mean(q_values)) if q_values else 0.0,
        "median_attack_raw_norm": float(np.mean(raw_norms)) if raw_norms else 0.0,
        "median_attack_clipped_norm": float(np.mean(norms)) if len(norms) else 0.0,
        "median_attack_was_clipped": float(np.mean(clipped_flags)) if clipped_flags else 0.0,
        "median_attack_clip_utilization": float(np.mean(norms) / max(max_norm, EPS)) if len(norms) else 0.0,
        "median_attack_mean_benign_norm": float(mean_norm),
        "median_attack_clip_norm": float(max_norm),
    }


def _median_state_from_context(
    ctx,
    *,
    previous_reward: float,
    previous_damage_gain: float,
    previous_alpha: float,
    max_alpha: float,
    clip_norm: Optional[float],
) -> np.ndarray:
    old_vec, benign_deltas = _benign_matrix(ctx)
    del old_vec
    max_norm = _clip_norm_from_context(ctx, benign_deltas, configured=clip_norm)
    benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
    norms = np.linalg.norm(benign_clipped, axis=1)
    clean_median = np.median(benign_clipped, axis=0)
    benign_center = np.mean(benign_clipped, axis=0)
    damage_direction = _normalize(-benign_center)
    selected_count = float(len(ctx.selected_attacker_ids))
    sampled_count = float(len(ctx.selected_attacker_ids) + len(ctx.benign_weights))
    total_clients = float(max(1, getattr(getattr(ctx, "fl_config", None), "num_clients", 1)))
    state = np.asarray(
        [
            float(getattr(ctx, "round_idx", 0)) / 1000.0,
            float(np.mean(norms)) / max(max_norm, EPS),
            float(np.std(norms)) / max(max_norm, EPS),
            float(np.linalg.norm(clean_median)) / max(max_norm, EPS),
            float(np.dot(clean_median, damage_direction)) / max(max_norm, EPS),
            selected_count / total_clients,
            selected_count / max(sampled_count, 1.0),
            previous_reward / 5.0,
            previous_damage_gain / max(max_norm, EPS),
            previous_alpha / max(max_alpha, EPS),
            max_norm / 10.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)


@dataclass
class ClippedMedianGeometrySearchConfig:
    max_alpha: float = 6.0
    search_steps: int = 32
    coordinate_std_floor: float = 1e-6
    direction_mix: float = 0.25
    clip_norm: Optional[float] = None


class ClippedMedianGeometrySearchAttack(SandboxAttack):
    """Coordinate-rank geometry-search attack for clipped median.

    Clipped median has no selected update, so this attacker searches for a
    malicious vector whose clipped copy shifts the coordinate-wise median
    toward the damage direction.
    """

    name = "ClippedMedianGeometrySearch"
    attack_type = "clipped_median_geometry_search"

    def __init__(self, config: Optional[ClippedMedianGeometrySearchConfig] = None) -> None:
        self.config = config or ClippedMedianGeometrySearchConfig()
        self.last_action_metrics: Dict[str, float] = {}

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0 or not ctx.benign_weights:
            return self.fallback_old_weights(ctx)

        old_vec, benign_deltas = _benign_matrix(ctx)
        max_norm = _clip_norm_from_context(ctx, benign_deltas, configured=self.config.clip_norm)
        benign_clipped = _clip_rows(benign_deltas, max_norm=max_norm)
        benign_center = np.mean(benign_clipped, axis=0)
        damage_direction = _normalize(-benign_center)
        if not np.any(damage_direction):
            damage_direction = _normalize(-np.mean(benign_deltas, axis=0))
        if not np.any(damage_direction):
            return self.fallback_old_weights(ctx)

        coord_median = np.median(benign_clipped, axis=0)
        coord_std = np.std(benign_clipped, axis=0)
        std_floor = max(float(self.config.coordinate_std_floor), EPS)
        coord_scale = np.maximum(coord_std, std_floor)
        mean_norm = float(np.mean(np.linalg.norm(benign_clipped, axis=1)))
        sign_direction = np.sign(damage_direction)
        dense_direction = damage_direction * max(mean_norm, EPS)

        best_delta = None
        best_score = -float("inf")
        best_alpha = 0.0
        best_metrics: Dict[str, float] = {}
        alphas = np.linspace(0.0, float(self.config.max_alpha), max(2, int(self.config.search_steps)))
        for alpha in alphas:
            rank_candidate = coord_median + float(alpha) * coord_scale * sign_direction
            candidate = (
                (1.0 - float(self.config.direction_mix)) * rank_candidate
                + float(self.config.direction_mix) * (coord_median + float(alpha) * dense_direction)
            )
            candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
            candidate_clipped, _, _ = _clip_one(candidate, max_norm=max_norm)
            attacked_median = _median_with_identical_attack(
                benign_clipped,
                candidate_clipped,
                num_attackers=num_attackers,
            )
            clean_median = np.median(benign_clipped, axis=0)
            shift = attacked_median - clean_median
            damage_gain = float(np.dot(shift, damage_direction))
            attacked_alignment = float(np.dot(attacked_median, damage_direction))
            norm_penalty = 0.02 * float(np.linalg.norm(candidate_clipped)) / max(max_norm, EPS)
            score = damage_gain + 0.1 * attacked_alignment - norm_penalty
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_delta = candidate_clipped
                best_metrics = clipped_median_shift_stats(
                    benign_deltas,
                    candidate,
                    num_attackers=num_attackers,
                    max_norm=max_norm,
                )

        if best_delta is None:
            return self.fallback_old_weights(ctx)

        crafted = vector_to_weights(old_vec + best_delta, ctx.old_weights)
        self.last_action_metrics = {
            **best_metrics,
            "median_geometry_alpha": float(best_alpha),
            "median_geometry_score": float(best_score),
            "median_geometry_clip_norm": float(max_norm),
            "median_geometry_num_attackers": float(num_attackers),
            "median_geometry_mean_benign_norm": float(mean_norm),
        }
        return [
            [np.asarray(layer, dtype=np.asarray(ctx.old_weights[idx]).dtype).copy() for idx, layer in enumerate(crafted)]
            for _ in range(num_attackers)
        ]

    def after_round(self, **kwargs):
        clean_loss_before = float(kwargs.get("clean_loss_before", float("nan")))
        clean_loss = float(kwargs.get("clean_loss", float("nan")))
        clean_acc_before = float(kwargs.get("clean_acc_before", float("nan")))
        clean_acc = float(kwargs.get("clean_acc", float("nan")))
        metrics = dict(self.last_action_metrics)
        if np.isfinite(clean_loss_before) and np.isfinite(clean_loss):
            metrics["median_geometry_clean_loss_delta"] = clean_loss - clean_loss_before
        if np.isfinite(clean_acc_before) and np.isfinite(clean_acc):
            metrics["median_geometry_clean_acc_delta"] = clean_acc - clean_acc_before
        return metrics

    def _clip_norm(self, ctx, benign_deltas: np.ndarray) -> float:
        return _clip_norm_from_context(ctx, benign_deltas, configured=self.config.clip_norm)


__all__ = [
    "ClippedMedianGeometrySearchAttack",
    "ClippedMedianGeometrySearchConfig",
    "clipped_median_shift_stats",
]
