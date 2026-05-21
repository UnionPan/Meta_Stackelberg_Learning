"""Configuration for the TD3-backed RL backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BackdoorRLConfig:
    """Paper-style TD3 configuration for adaptive backdoor policy learning."""

    algorithm: str = "td3"
    seed: int = 42
    action_dim: int = 4
    projection_dim: int = 256
    history_window: int = 1
    train_horizon: int = 20
    train_steps: int = 500
    train_freq_steps: int = 1
    replay_capacity: int = 50_000
    batch_size: int = 128
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    exploration_noise: float = 0.15
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    update_actor_freq: int = 2
    gradient_clip_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    recency_tau: float = 48.0
    reward_clean_weight: float = 1.0
    reward_norm_weight: float = 0.05
    policy_checkpoint_path: str = ""
    policy_checkpoint_dir: str = ""

    # PPO fields are present so the shared trainer protocol can still be used
    # if experiments explicitly switch algorithms later.
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gae_lambda: float = 0.95
    ppo_max_grad_norm: float = 0.5

    @property
    def per_step_observation_dim(self) -> int:
        return 2 * int(self.projection_dim) + int(self.action_dim) + 5

    @property
    def observation_dim(self) -> int:
        return int(self.per_step_observation_dim) * max(1, int(self.history_window))

    @property
    def action_low(self) -> np.ndarray:
        return -np.ones(int(self.action_dim), dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.ones(int(self.action_dim), dtype=np.float32)

    def policy_train_gradient_steps(self) -> int:
        return max(1, int(self.train_steps) // max(1, int(self.train_freq_steps)))

    def checkpoint_for_round(self, round_idx: int) -> str:
        if not self.policy_checkpoint_dir:
            return str(self.policy_checkpoint_path or "")
        directory = Path(self.policy_checkpoint_dir)
        if not directory.is_dir():
            return ""
        best_round = -1
        best_path: Path | None = None
        prefix = "rl_backdoor_td3_round_"
        for path in directory.glob(f"{prefix}*.pt"):
            suffix = path.stem.removeprefix(prefix)
            if suffix == path.stem:
                continue
            try:
                checkpoint_round = int(suffix)
            except ValueError:
                continue
            if checkpoint_round <= int(round_idx) and checkpoint_round > best_round:
                best_round = checkpoint_round
                best_path = path
        return str(best_path) if best_path is not None else ""

    @classmethod
    def from_attacker_config(cls, attacker_config) -> "BackdoorRLConfig":
        return cls(
            algorithm=str(getattr(attacker_config, "rl_algorithm", "td3")),
            policy_lr=float(getattr(attacker_config, "rl_policy_lr", 3e-4)),
            critic_lr=float(getattr(attacker_config, "rl_critic_lr", 3e-4)),
            gamma=float(getattr(attacker_config, "rl_gamma", 0.95)),
            replay_capacity=int(getattr(attacker_config, "rl_replay_capacity", 50_000)),
            batch_size=int(getattr(attacker_config, "rl_batch_size", 128)),
            hidden_sizes=tuple(getattr(attacker_config, "rl_hidden_sizes", (256, 256))),
            exploration_noise=float(getattr(attacker_config, "rl_exploration_noise", 0.15)),
            train_freq_steps=int(getattr(attacker_config, "rl_train_freq_steps", 1)),
            policy_checkpoint_path=str(getattr(attacker_config, "rl_policy_checkpoint_path", "")),
            policy_checkpoint_dir=str(getattr(attacker_config, "rl_policy_checkpoint_dir", "")),
        )
