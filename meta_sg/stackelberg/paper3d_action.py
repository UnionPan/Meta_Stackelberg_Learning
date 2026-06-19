"""Paper-aligned three-dimensional defender action helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Paper3DAction:
    alpha: float
    beta: float
    epsilon: float


def raw_action_to_paper3d(raw_action: np.ndarray, args: argparse.Namespace) -> Paper3DAction:
    raw = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if raw.shape[0] < 3:
        raw = np.pad(raw, (0, 3 - raw.shape[0]), constant_values=-1.0)
    return Paper3DAction(
        alpha=_scale_raw(raw[0], float(args.alpha_min), float(args.alpha_max)),
        beta=_scale_raw(raw[1], float(args.beta_min), float(args.beta_max)),
        epsilon=_scale_raw(raw[2], *_third_action_range(args)),
    )


def fixed_paper3d_to_raw_action(action: Paper3DAction, args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [
            _unscale_raw(action.alpha, float(args.alpha_min), float(args.alpha_max)),
            _unscale_raw(action.beta, float(args.beta_min), float(args.beta_max)),
            _unscale_raw(action.epsilon, *_third_action_range(args)),
        ],
        dtype=np.float32,
    )


def parse_fixed_paper3d_baselines(values: list[str] | None) -> dict[str, Paper3DAction]:
    if not values:
        values = [
            "weak_loose=30,0,10",
            "norm_only_mid=4,0,10",
            "trim_only=30,0.2,10",
            "norm_trim=4,0.2,10",
            "norm_trim_neuro2=4,0.2,2",
            "norm_trim_neuro5=4,0.2,5",
        ]
    result: dict[str, Paper3DAction] = {}
    for item in values:
        name, raw = item.split("=", 1)
        alpha, beta, epsilon = [float(part.strip()) for part in raw.split(",")]
        result[name.strip()] = Paper3DAction(alpha=alpha, beta=beta, epsilon=epsilon)
    return result


def _scale_raw(raw: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError(f"invalid range: high={high} must be greater than low={low}")
    return float(low + (float(raw) + 1.0) * 0.5 * (high - low))


def _third_action_range(args: argparse.Namespace) -> tuple[float, float]:
    if str(getattr(args, "third_action", "neuroclip")) == "server_lr":
        return float(args.server_lr_min), float(args.server_lr_max)
    return float(args.neuroclip_eps_min), float(args.neuroclip_eps_max)


def _unscale_raw(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError(f"invalid range: high={high} must be greater than low={low}")
    raw = 2.0 * (float(value) - low) / (high - low) - 1.0
    return float(np.clip(raw, -1.0, 1.0))
