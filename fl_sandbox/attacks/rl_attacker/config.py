"""Configuration for the Tianshou-backed RL attacker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_CLIPPED_MEDIAN = "legacy_clipped_median"
LEGACY_CLIPPED_MEDIAN_STRICT = "legacy_clipped_median_strict"
LEGACY_CLIPPED_MEDIAN_SCALEAWARE = "legacy_clipped_median_scaleaware"
LEGACY_CLIPPED_MEDIAN_SEMANTICS = frozenset(
    {LEGACY_CLIPPED_MEDIAN, LEGACY_CLIPPED_MEDIAN_STRICT, LEGACY_CLIPPED_MEDIAN_SCALEAWARE}
)
LEGACY_KRUM_STRICT = "legacy_krum_strict"
LEGACY_KRUM_GEOMETRY = "legacy_krum_geometry"
LEGACY_KRUM_SEMANTICS = frozenset({LEGACY_KRUM_STRICT, LEGACY_KRUM_GEOMETRY})


@dataclass
class RLAttackerConfig:
    """Config for proxy learning, simulation, and Tianshou policy training."""

    algorithm: str = "td3"
    attacker_semantics: str = "canonical"
    seed: int = 42
    distribution_steps: int = 10
    attack_start_round: int = 10
    policy_train_end_round: int = 30
    inversion_lr: float = 0.05
    inversion_steps: int = 50
    reconstruction_batch_size: int = 8
    reconstruction_quality_threshold: float = 0.7
    tv_weight: float = 0.02
    denoiser_noise_std: float = 0.3
    denoiser_epochs: int = 2
    seed_samples: int = 128
    proxy_buffer_limit: int = 2048
    state_tail_layers: int = 2
    state_include_num_attacker: bool = True
    projection_dim: int = 256
    history_window: int = 4
    simulator_horizon: int = 12
    simulator_refresh_interval: int = 10
    episodes_per_observation: int = 2
    policy_train_steps_per_round: int = 0
    replay_capacity: int = 50_000
    batch_size: int = 256
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    update_actor_freq: int = 2
    train_freq_steps: int = 1
    gradient_clip_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    policy_checkpoint_path: str = ""
    policy_checkpoint_dir: str = ""
    freeze_policy: bool = False
    warmup_steps: int = 1000
    update_to_data_ratio: int = 1
    recency_tau: float = 48.0
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gae_lambda: float = 0.95
    ppo_max_grad_norm: float = 0.5
    ppo_real_rollout_steps: int = 64
    hybrid_template_dim: int = 5
    hybrid_continuous_dim: int = 7
    local_search_batch_size: int = 200
    attacker_local_lr: float = 0.05
    reward_loss_weight: float = 1.0
    reward_accuracy_weight: float = 1.0
    reward_norm_penalty_weight: float = 0.0
    reward_bypass_weight: float = 0.5
    reward_action_smoothness_weight: float = 0.01
    reward_action_saturation_weight: float = 0.1
    reward_template_switch_weight: float = 0.1
    robust_gamma_center: float = 5.0
    robust_gamma_scale: float = 4.9
    robust_steps_center: float = 11.0
    robust_steps_scale: float = 10.0
    clipped_gamma_center: float = 15.0
    clipped_gamma_scale: float = 14.9
    clipped_steps_center: float = 25.0
    clipped_steps_scale: float = 24.0
    legacy_clipped_median_lr: float = 0.01
    legacy_krum_gamma_center: float = 5.0
    legacy_krum_gamma_scale: float = 4.9
    legacy_krum_steps_center: float = 11.0
    legacy_krum_steps_scale: float = 10.0
    legacy_krum_lr: float = 0.01
    strict_reproduction_initial_samples: int = 200
    strict_reproduction_samples_per_epoch: int = 80
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
    deploy_guard_window: int = 20

    def uses_legacy_clipped_median(self) -> bool:
        return self.attacker_semantics.lower() in LEGACY_CLIPPED_MEDIAN_SEMANTICS

    def uses_legacy_krum(self) -> bool:
        return self.attacker_semantics.lower() in LEGACY_KRUM_SEMANTICS

    def uses_legacy_krum_geometry(self) -> bool:
        return self.attacker_semantics.lower() == LEGACY_KRUM_GEOMETRY

    def uses_scaleaware_legacy_observation(self) -> bool:
        return self.attacker_semantics.lower() == LEGACY_CLIPPED_MEDIAN_SCALEAWARE

    def uses_legacy_reversal_attack(self) -> bool:
        return self.uses_legacy_clipped_median() or self.uses_legacy_krum()

    def uses_raw_loss_delta_reward(self) -> bool:
        return self.uses_legacy_reversal_attack()

    def uses_strict_reproduction(self) -> bool:
        return self.attacker_semantics.lower() in {
            LEGACY_CLIPPED_MEDIAN_STRICT,
            LEGACY_CLIPPED_MEDIAN_SCALEAWARE,
            LEGACY_KRUM_STRICT,
            LEGACY_KRUM_GEOMETRY,
        }

    def strict_reproduction_sample_limit(self, *, epoch: int, buffer_size: int) -> int:
        if buffer_size <= 0:
            return 0
        epoch = max(1, int(epoch))
        initial = max(1, int(self.strict_reproduction_initial_samples))
        increment = max(0, int(self.strict_reproduction_samples_per_epoch))
        return min(int(buffer_size), initial + (epoch - 1) * increment)

    def policy_train_steps(self) -> int:
        if int(self.policy_train_steps_per_round) > 0:
            return max(1, int(self.policy_train_steps_per_round))
        return max(1, int(self.episodes_per_observation)) * max(1, int(self.simulator_horizon))

    def has_policy_checkpoint_source(self) -> bool:
        return bool(self.policy_checkpoint_path or self.policy_checkpoint_dir)

    def policy_checkpoint_for_round(self, round_idx: int) -> str:
        if self.policy_checkpoint_dir:
            return self._rolling_policy_checkpoint_for_round(round_idx)
        return str(self.policy_checkpoint_path or "")

    def _rolling_policy_checkpoint_for_round(self, round_idx: int) -> str:
        directory = Path(self.policy_checkpoint_dir)
        if not directory.is_dir():
            return ""
        cap_round = max(0, int(round_idx))
        if int(self.policy_train_end_round or 0) > 0:
            cap_round = min(cap_round, int(self.policy_train_end_round))

        best_round = -1
        best_path: Path | None = None
        prefix = "rl_policy_round_"
        for path in directory.glob(f"{prefix}*.pt"):
            suffix = path.stem.removeprefix(prefix)
            if suffix == path.stem:
                continue
            try:
                checkpoint_round = int(suffix)
            except ValueError:
                continue
            if checkpoint_round <= cap_round and checkpoint_round > best_round:
                best_round = checkpoint_round
                best_path = path
        return str(best_path) if best_path is not None else ""

    def action_dim(self, defense_type: str) -> int:
        del defense_type
        if self.uses_legacy_reversal_attack():
            return 2
        if self.algorithm.lower() == "ppo":
            return 1 + int(self.hybrid_continuous_dim)
        return 3

    def validate_defense(self, defense_type: str) -> None:
        defense = defense_type.lower()
        if self.uses_legacy_clipped_median() and defense != "clipped_median":
            raise ValueError(
                "legacy_clipped_median attacker semantics only support defense_type='clipped_median'"
            )
        if self.uses_legacy_krum() and defense != "krum":
            raise ValueError("legacy_krum attacker semantics only support defense_type='krum'")

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
