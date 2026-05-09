"""Defense-aware action decoding for the RL attacker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig

KRUM_DEFENSES = frozenset({"krum", "multi_krum"})
CLIPMED_DEFENSES = frozenset({"clipped_median"})
COORD_MEDIAN_DEFENSES = frozenset({"median", "trimmed_mean", "geometric_median", "paper_norm_trimmed_mean"})


@dataclass(frozen=True)
class AttackParameters:
    gamma_scale: float
    local_steps: int
    lambda_stealth: float
    local_search_lr: float


DecodedAction = AttackParameters


def normalized_action(action: np.ndarray, dim: int = 3) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32)
    action_arr = np.nan_to_num(action_arr, nan=0.0, posinf=1.0, neginf=-1.0).reshape(-1)
    if action_arr.shape[0] < dim:
        padded = np.zeros(dim, dtype=np.float32)
        padded[: action_arr.shape[0]] = action_arr
        action_arr = padded
    return np.clip(action_arr[:dim], -1.0, 1.0)


def decode_action(action: np.ndarray, defense_type: str, config: RLAttackerConfig) -> AttackParameters:
    action_arr = normalized_action(action, config.action_dim(defense_type))
    defense = defense_type.lower()
    if defense in KRUM_DEFENSES:
        gamma_scale = float(action_arr[0]) * config.krum_gamma_scale + config.krum_gamma_center
        local_steps = int(round(float(action_arr[1]) * config.krum_steps_scale + config.krum_steps_center))
        lambda_stealth = float(action_arr[2]) * config.krum_stealth_scale + config.krum_stealth_center
    elif defense in CLIPMED_DEFENSES:
        gamma_scale = float(action_arr[0]) * config.clipmed_gamma_scale + config.clipmed_gamma_center
        local_steps = int(round(float(action_arr[1]) * config.clipmed_steps_scale + config.clipmed_steps_center))
        lambda_stealth = float(action_arr[2]) * config.clipmed_lambda_scale + config.clipmed_lambda_center
    elif defense in COORD_MEDIAN_DEFENSES:
        gamma_scale = float(action_arr[0]) * config.coord_gamma_scale + config.coord_gamma_center
        local_steps = int(round(float(action_arr[1]) * config.coord_steps_scale + config.coord_steps_center))
        lambda_stealth = float(action_arr[2]) * config.coord_lambda_scale + config.coord_lambda_center
    elif defense == "fltrust":
        gamma_scale = 1.0
        local_steps = int(round(float(action_arr[1]) * config.robust_steps_scale + config.robust_steps_center))
        lambda_stealth = float(action_arr[2]) * config.fltrust_alpha_scale + config.fltrust_alpha_center
        local_search_lr = float(action_arr[0]) * config.fltrust_lr_scale + config.fltrust_lr_center
        return AttackParameters(
            gamma_scale=gamma_scale,
            local_steps=max(1, local_steps),
            lambda_stealth=float(np.clip(lambda_stealth, 0.0, 1.0)),
            local_search_lr=max(1e-4, float(local_search_lr)),
        )
    else:
        gamma_scale = float(action_arr[0]) * config.robust_gamma_scale + config.robust_gamma_center
        local_steps = int(round(float(action_arr[1]) * config.robust_steps_scale + config.robust_steps_center))
        lambda_stealth = 0.0
    return AttackParameters(
        gamma_scale=max(0.1, float(gamma_scale)),
        local_steps=max(1, local_steps),
        lambda_stealth=float(np.clip(lambda_stealth, 0.0, 1.0)),
        local_search_lr=max(1e-4, float(config.attacker_local_lr)),
    )
