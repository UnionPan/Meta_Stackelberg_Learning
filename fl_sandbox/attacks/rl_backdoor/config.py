"""Configuration for the TD3-backed RL backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BackdoorRLConfig:
    """Paper-style TD3 configuration for adaptive backdoor policy learning.

    Two distinct tau values intentionally share related names — ``tau`` is the
    TD3 target-network update rate, ``round_phase_tau`` is the time-constant of
    the ``tanh(round / tau)`` state component. Don't unify them.
    """

    algorithm: str = "td3"
    seed: int = 42
    action_dim: int = 4
    projection_dim: int = 128
    history_window: int = 1
    state_tail_layers: int = 2
    round_phase_tau: float = 100.0

    # TD3 training
    train_horizon: int = 500
    train_steps: int = 500
    train_freq_steps: int = 1
    replay_capacity: int = 200_000
    batch_size: int = 128
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 1.0
    tau: float = 0.005
    exploration_noise: float = 0.05
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    update_actor_freq: int = 2
    gradient_clip_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    recency_tau: float = 48.0

    # Grey-box simulator
    defense_mode: str = "known"  # {"known", "randomized"}
    simulator_lr: float = 0.01
    simulator_local_epochs: int = 1
    # 0 → fall back to the live FL ``num_clients`` (no sub-partition override).
    # >0 → sub-partition the attacker's data into this many shadow benign
    # client shards, with ``samples_per_client`` samples each.
    simulator_shadow_clients: int = 10
    simulator_shadow_samples_per_client: int = 200

    # Reward
    reward_mode: str = "paper"  # {"paper"/"henger_li", "delta"/"asr_delta", "stealth"}
    reward_clean_lambda: float = 0.5  # paper mode: lambda in F' = lambda*F(U)+(1-lambda)*F(U')
    reward_clean_weight: float = 1.0  # only consulted when reward_mode == "stealth"
    reward_norm_weight: float = 0.1   # only consulted when reward_mode == "stealth"

    # When ``stealth_norm_cap`` is on, the malicious update's norm is hard-clamped
    # to the benign mean, which saturates the boost action dim — the policy has
    # nothing left to learn on a[3]. Pinning ``freeze_boost`` to a known-good
    # scalar (e.g. 5.0, matching the static-action baseline) frees the actor's
    # capacity for the remaining 3 dims (poison rate, lr, epochs) and lets RL
    # at least match the strong fixed-action baseline instead of being stuck
    # learning a dead dimension. Set to ``None`` to let the policy learn boost.
    freeze_boost: Optional[float] = 5.0
    # MUST match the live-path cap setting — see ``RLBackdoorAttack._craft_sybil_broadcast``.
    # If live applies a benign-norm cap and sim doesn't, the policy trains on
    # one Krum dynamics and deploys against another (sim→real mismatch). The
    # cap is only meaningful when the defender filters by norm; ``fedavg`` is
    # treated as a no-op identically on both sides.
    stealth_norm_cap: bool = False
    # Warm-start the replay buffer by rolling out the static default action in
    # the simulator for ``warmup_fixed_rollouts`` steps before the first TD3
    # update. Without this the policy spends the early training rounds with
    # random actions, never plants the backdoor, gets zero reward, and the
    # gradient signal collapses. 0 disables.
    warmup_fixed_rollouts: int = 200

    # Checkpoint / deployment
    policy_checkpoint_path: str = ""
    policy_checkpoint_dir: str = ""
    freeze_policy: bool = False

    # PPO fields kept so build_trainer can dispatch if explicitly switched
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gae_lambda: float = 0.95
    ppo_max_grad_norm: float = 0.5

    @property
    def per_step_observation_dim(self) -> int:
        # round_phase + att_frac_atk + att_frac_cli + local_bd_success
        # + delta_lognorm + proj(tail dir) + proj(delta tail dir) + last_action
        return 5 + 2 * int(self.projection_dim) + int(self.action_dim)

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
        for prefix in ("rl_policy_round_", "rl_backdoor_td3_round_"):
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
        freeze_boost_raw = getattr(attacker_config, "rl_backdoor_freeze_boost", 5.0)
        freeze_boost = None if freeze_boost_raw is None else float(freeze_boost_raw)
        return cls(
            algorithm=str(getattr(attacker_config, "rl_algorithm", "td3")),
            policy_lr=float(getattr(attacker_config, "rl_policy_lr", 3e-4)),
            critic_lr=float(getattr(attacker_config, "rl_critic_lr", 3e-4)),
            gamma=float(getattr(attacker_config, "rl_gamma", 1.0)),
            train_horizon=max(1, int(getattr(attacker_config, "rl_simulator_horizon", 500))),
            replay_capacity=int(getattr(attacker_config, "rl_replay_capacity", 200_000)),
            batch_size=int(getattr(attacker_config, "rl_batch_size", 128)),
            hidden_sizes=tuple(getattr(attacker_config, "rl_hidden_sizes", (256, 256))),
            exploration_noise=float(getattr(attacker_config, "rl_exploration_noise", 0.05)),
            train_freq_steps=int(getattr(attacker_config, "rl_train_freq_steps", 1)),
            policy_checkpoint_path=str(getattr(attacker_config, "rl_policy_checkpoint_path", "")),
            policy_checkpoint_dir=str(getattr(attacker_config, "rl_policy_checkpoint_dir", "")),
            freeze_policy=bool(getattr(attacker_config, "rl_freeze_policy", False)),
            reward_mode=str(getattr(attacker_config, "rl_backdoor_reward_mode", "paper")),
            reward_clean_lambda=float(
                getattr(attacker_config, "rl_backdoor_reward_clean_lambda", 0.5)
            ),
            reward_clean_weight=float(
                getattr(attacker_config, "rl_backdoor_reward_clean_lambda", 1.0)
            ),
            reward_norm_weight=float(
                getattr(attacker_config, "rl_backdoor_reward_norm_lambda", 0.1)
            ),
            freeze_boost=freeze_boost,
            stealth_norm_cap=bool(
                getattr(attacker_config, "rl_backdoor_stealth_norm_cap", False)
            ),
            warmup_fixed_rollouts=int(
                getattr(attacker_config, "rl_backdoor_warmup_fixed_rollouts", 200)
            ),
            simulator_shadow_clients=int(
                getattr(attacker_config, "rl_backdoor_simulator_shadow_clients", 10)
            ),
            simulator_shadow_samples_per_client=int(
                getattr(attacker_config, "rl_backdoor_simulator_shadow_samples_per_client", 200)
            ),
        )
