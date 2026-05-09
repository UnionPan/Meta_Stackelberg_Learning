"""Configuration for the Tianshou-backed RL attacker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RLAttackerConfig:
    """Config for proxy learning, simulation, and Tianshou policy training."""

    algorithm: str = "sac"
    distribution_steps: int = 10
    attack_start_round: int = 10
    policy_train_end_round: int = 30
    inversion_lr: float = 0.05
    inversion_steps: int = 50
    reconstruction_batch_size: int = 8
    reconstruction_quality_threshold: float = -1.0
    tv_weight: float = 0.02
    denoiser_noise_std: float = 0.3
    denoiser_epochs: int = 2
    seed_samples: int = 128
    proxy_buffer_limit: int = 2048
    state_tail_layers: int = 2
    state_include_num_attacker: bool = True
    history_window: int = 4
    simulator_horizon: int = 10
    episodes_per_observation: int = 2
    replay_capacity: int = 50_000
    batch_size: int = 64
    policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    gamma: float = 1.0
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    update_actor_freq: int = 2
    train_freq_steps: int = 1
    gradient_clip_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    local_search_batch_size: int = 200
    attacker_local_lr: float = 0.05
    reward_accuracy_weight: float = 6.0
    reward_norm_penalty_weight: float = 0.75
    reward_bypass_weight: float = 0.5
    reward_action_smoothness_weight: float = 0.02
    reward_action_saturation_weight: float = 0.02
    robust_gamma_center: float = 5.0
    robust_gamma_scale: float = 4.9
    robust_steps_center: float = 11.0
    robust_steps_scale: float = 10.0
    clipped_gamma_center: float = 15.0
    clipped_gamma_scale: float = 14.9
    clipped_steps_center: float = 25.0
    clipped_steps_scale: float = 24.0
    fltrust_lr_center: float = 0.05
    fltrust_lr_scale: float = 0.04
    fltrust_alpha_center: float = 0.5
    fltrust_alpha_scale: float = 0.5
    krum_gamma_center: float = 1.5
    krum_gamma_scale: float = 1.4
    krum_steps_center: float = 10.0
    krum_steps_scale: float = 9.0
    krum_stealth_center: float = 0.5
    krum_stealth_scale: float = 0.45
    clipmed_gamma_center: float = 1.2
    clipmed_gamma_scale: float = 1.1
    clipmed_steps_center: float = 10.0
    clipmed_steps_scale: float = 9.0
    clipmed_lambda_center: float = 0.20
    clipmed_lambda_scale: float = 0.20
    coord_gamma_center: float = 8.0
    coord_gamma_scale: float = 7.9
    coord_steps_center: float = 18.0
    coord_steps_scale: float = 17.0
    coord_lambda_center: float = 0.30
    coord_lambda_scale: float = 0.20
    clipmed_target_norm_ratio: float = 1.25
    coord_target_norm_ratio: float = 1.10
    krum_target_norm_ratio: float = 1.05
    fltrust_target_norm_ratio: float = 1.50
    deploy_guard_min_proxy_samples: int = 8
    deploy_guard_max_sim2real_gap: float = 5.0

    def action_dim(self, defense_type: str) -> int:
        return 3

    def action_bounds(self, defense_type: str) -> tuple[np.ndarray, np.ndarray]:
        dim = self.action_dim(defense_type)
        return -np.ones(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)

    def target_norm_ratio(self, defense_type: str) -> float:
        defense = defense_type.lower()
        if defense == "clipped_median":
            return self.clipmed_target_norm_ratio
        if defense in {"median", "trimmed_mean", "geometric_median", "paper_norm_trimmed_mean"}:
            return self.coord_target_norm_ratio
        if defense in {"krum", "multi_krum"}:
            return self.krum_target_norm_ratio
        if defense == "fltrust":
            return self.fltrust_target_norm_ratio
        return self.coord_target_norm_ratio
