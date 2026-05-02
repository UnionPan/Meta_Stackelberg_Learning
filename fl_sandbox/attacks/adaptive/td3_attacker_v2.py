"""RL attacker v2: defense-aware policy with aggressive paper-style robust crafting.

Improvements over v1:
1. Krum-family defenses use stealth-aware local search with explicit bypass reward.
2. Coordinate-wise robust defenses default to a stronger paper-style 2-D attack space.
3. Robust attack ranges are widened toward the paper's clipped-median operating regime.
4. policy_lr = 1e-4 — much faster than v1 for online sandbox training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from fl_sandbox.attacks.adaptive.td3_attacker import (
    PaperRLAttacker,
    RLAttackerConfig,
    SimulatedFLEnv,
    DecodedAction,
    ReplayBuffer,
    TD3Agent,
    local_search_update,
    _train_proxy_model,
)

_KRUM_DEFENSES = frozenset({"krum", "multi_krum"})
_AGGRESSIVE_ROBUST_DEFENSES = frozenset({"median", "clipped_median", "trimmed_mean", "geometric_median"})
_STEALTH_DEFENSES = frozenset({"krum", "multi_krum"})


@dataclass
class RLAttackerConfigV2(RLAttackerConfig):
    """V2 config — faster learning, stealth-aware robust action space, bypass reward."""

    simulator_horizon: int = 10
    episodes_per_observation: int = 2
    td3_batch_size: int = 64
    td3_train_freq_steps: int = 1
    policy_lr: float = 1e-4
    bypass_reward_weight: float = 0.5
    krum_gamma_center: float = 1.5
    krum_gamma_scale: float = 1.4
    krum_steps_center: float = 10.0
    krum_steps_scale: float = 9.0
    krum_stealth_center: float = 0.5
    krum_stealth_scale: float = 0.45
    stealth_gamma_center: float = 1.2
    stealth_gamma_scale: float = 1.1
    stealth_steps_center: float = 10.0
    stealth_steps_scale: float = 9.0
    stealth_lambda_center: float = 0.5
    stealth_lambda_scale: float = 0.45
    aggressive_gamma_center: float = 15.0
    aggressive_gamma_scale: float = 14.9
    aggressive_steps_center: float = 25.0
    aggressive_steps_scale: float = 24.0

    def action_dim(self, defense_type: str) -> int:
        defense = defense_type.lower()
        if defense in _AGGRESSIVE_ROBUST_DEFENSES:
            return 2
        if defense == "fltrust":
            return super().action_dim(defense_type)
        if defense in _STEALTH_DEFENSES:
            return 3
        return super().action_dim(defense_type)

    def decode_action(self, action: np.ndarray, defense_type: str) -> DecodedAction:
        defense = defense_type.lower()
        if defense == "fltrust":
            return super().decode_action(action, defense_type)
        if defense in _AGGRESSIVE_ROBUST_DEFENSES:
            action_arr = np.asarray(action, dtype=np.float32)
            if action_arr.shape[0] < 2:
                padded = np.zeros(2, dtype=np.float32)
                padded[: action_arr.shape[0]] = action_arr
                action_arr = padded
            action_arr = np.clip(action_arr[:2], -1.0, 1.0)
            gamma_scale = float(action_arr[0]) * self.aggressive_gamma_scale + self.aggressive_gamma_center
            local_steps = int(round(float(action_arr[1]) * self.aggressive_steps_scale + self.aggressive_steps_center))
            return DecodedAction(
                gamma_scale=max(0.1, gamma_scale),
                local_steps=max(1, local_steps),
                lambda_stealth=0.0,
                local_search_lr=self.attacker_local_lr,
            )
        if defense not in _STEALTH_DEFENSES:
            return super().decode_action(action, defense_type)
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape[0] < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: action_arr.shape[0]] = action_arr
            action_arr = padded
        action_arr = np.clip(action_arr[:3], -1.0, 1.0)
        if defense in _KRUM_DEFENSES:
            gamma_center = self.krum_gamma_center
            gamma_scale_span = self.krum_gamma_scale
            steps_center = self.krum_steps_center
            steps_scale = self.krum_steps_scale
            stealth_center = self.krum_stealth_center
            stealth_scale = self.krum_stealth_scale
        else:
            gamma_center = self.stealth_gamma_center
            gamma_scale_span = self.stealth_gamma_scale
            steps_center = self.stealth_steps_center
            steps_scale = self.stealth_steps_scale
            stealth_center = self.stealth_lambda_center
            stealth_scale = self.stealth_lambda_scale
        gamma_scale = float(action_arr[0]) * gamma_scale_span + gamma_center
        local_steps = int(round(float(action_arr[1]) * steps_scale + steps_center))
        lambda_stealth = float(action_arr[2]) * stealth_scale + stealth_center
        return DecodedAction(
            gamma_scale=max(0.1, gamma_scale),
            local_steps=max(1, local_steps),
            lambda_stealth=float(np.clip(lambda_stealth, 0.0, 1.0)),
            local_search_lr=self.attacker_local_lr,
        )


class SimulatedFLEnvV2(SimulatedFLEnv):
    """V2 simulator: stealth-craft for robust defenses + bypass reward for krum."""

    def step(self, action: np.ndarray):
        if self.current_weights is None:
            raise RuntimeError("SimulatedFLEnvV2 must be reset before stepping")
        benign_count = max(
            0,
            max(1, int(self.fl_config.num_clients * self.fl_config.subsample_rate)) - self.current_num_attackers,
        )
        benign_weights = [self._simulate_benign_update(self.current_weights) for _ in range(benign_count)]
        malicious_weights = [
            self._simulate_malicious_weight(self.current_weights, action)
            for _ in range(self.current_num_attackers)
        ]
        bypass_reward = 0.0
        defense_type = self.defender.defense_type.lower()
        bypass_weight = float(getattr(self.config, "bypass_reward_weight", 0.0))
        if defense_type in _KRUM_DEFENSES and bypass_weight > 0:
            bypass_reward = self._compute_bypass_reward(benign_weights, malicious_weights)
        trusted_weights = self._trusted_reference_update(self.current_weights)
        updates = benign_weights + malicious_weights
        if updates:
            self.current_weights = self.defender.aggregate(self.current_weights, updates, trusted_weights=trusted_weights)
        new_loss = self._evaluate_proxy_loss(self.current_weights)
        loss_reward = new_loss - self.current_loss
        self.current_loss = new_loss
        self.round_idx += 1
        done = self.round_idx >= max(1, self.config.simulator_horizon)
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        bw = bypass_weight
        reward = (1.0 - bw) * loss_reward + bw * bypass_reward
        return self._get_state(), reward, done

    def _simulate_malicious_weight(self, old_weights, action):
        defense_type = self.defender.defense_type.lower()
        action_params = self.config.decode_action(action, defense_type)
        if defense_type in _STEALTH_DEFENSES or defense_type == "fltrust":
            return local_search_update(
                model_template=self.model_template,
                old_weights=old_weights,
                proxy_buffer=self.proxy_buffer,
                device=self.device,
                fl_lr=action_params.local_search_lr,
                steps=action_params.local_steps,
                gamma_scale=action_params.gamma_scale,
                lambda_stealth=action_params.lambda_stealth,
                search_batch_size=self.config.local_search_batch_size,
                state_tail_layers=self.config.state_tail_layers,
            )
        trained_weights = _train_proxy_model(
            model_template=self.model_template,
            old_weights=old_weights,
            proxy_buffer=self.proxy_buffer,
            device=self.device,
            lr=action_params.local_search_lr,
            steps=action_params.local_steps,
            batch_size=self.config.local_search_batch_size,
        )
        from fl_sandbox.attacks.vector import craft_ipm
        return craft_ipm(old_weights, trained_weights, scale=action_params.gamma_scale)

    def _compute_bypass_reward(self, benign_weights, malicious_weights) -> float:
        from fl_sandbox.aggregators.rules import _krum_candidate_indices, _stack_updates

        n_benign = len(benign_weights)
        all_updates = benign_weights + malicious_weights
        if len(all_updates) <= 1 or len(malicious_weights) == 0:
            return 0.0
        krum_attackers = max(0, getattr(self.fl_config, "krum_attackers", 1))
        try:
            update_matrix = _stack_updates(self.current_weights, all_updates)
            ranked = _krum_candidate_indices(update_matrix, num_attackers=krum_attackers)
        except Exception:
            return 0.0
        if self.defender.defense_type.lower() == "multi_krum":
            n = len(all_updates)
            max_f = max(0, (n - 3) // 2)
            f = min(krum_attackers, max_f)
            n_selected = max(1, n - f - 2)
        else:
            n_selected = 1
        selected = set(ranked[:n_selected])
        bypassed = any(idx >= n_benign for idx in selected)
        return 1.0 if bypassed else -0.1


class PaperRLAttackerV2(PaperRLAttacker):
    """V2 attacker: stealth craft + bypass reward + faster policy learning."""

    def __init__(self, config: Optional[RLAttackerConfigV2] = None) -> None:
        super().__init__(config or RLAttackerConfigV2())

    def _craft_malicious_weights(self, old_weights, decoded: DecodedAction, defense_type: str):
        if defense_type.lower() in _STEALTH_DEFENSES or defense_type.lower() == "fltrust":
            return local_search_update(
                model_template=self.model_template,
                old_weights=old_weights,
                proxy_buffer=self.distribution_learner.buffer,
                device=self.device,
                fl_lr=decoded.local_search_lr,
                steps=decoded.local_steps,
                gamma_scale=decoded.gamma_scale,
                lambda_stealth=decoded.lambda_stealth,
                search_batch_size=self.config.local_search_batch_size,
                state_tail_layers=self.config.state_tail_layers,
            )
        from fl_sandbox.attacks.vector import craft_ipm
        trained_weights = _train_proxy_model(
            model_template=self.model_template,
            old_weights=old_weights,
            proxy_buffer=self.distribution_learner.buffer,
            device=self.device,
            lr=decoded.local_search_lr,
            steps=decoded.local_steps,
            batch_size=self.config.local_search_batch_size,
        )
        return craft_ipm(old_weights, trained_weights, scale=decoded.gamma_scale)

    def _train_policy(self) -> None:
        if not self.ready or self.model_template is None or self.defender is None or self.fl_config is None:
            return
        if len(self.distribution_learner.buffer) < max(8, self.config.reconstruction_batch_size):
            return
        state_dim = self._infer_state_dim()
        low, high = self.config.action_bounds(self.defender.defense_type)
        if self.policy is None:
            self.policy = TD3Agent(state_dim, low, high, self.config, self.device)
        if self.latest_policy_weights is None:
            return
        from fl_sandbox.attacks.adaptive.pz_env import AttackerPolicyParallelEnv
        sim_env = SimulatedFLEnvV2(
            model_template=self.model_template,
            proxy_buffer=self.distribution_learner.buffer,
            defender=self.defender,
            config=self.config,
            fl_config=self.fl_config,
            device=self.device,
        )
        parallel_env = AttackerPolicyParallelEnv(
            simulator=sim_env,
            rl_config=self.config,
            defense_type=self.defender.defense_type,
            initial_weights=self.latest_policy_weights,
        )
        for _ in range(max(1, self.config.episodes_per_observation)):
            obs, _ = parallel_env.reset()
            state = self.config.flatten_state(obs["attacker"], sim_env.max_attackers_sampled)
            terminated = False
            truncated = False
            steps_since_update = 0
            while not (terminated or truncated):
                if len(self.replay) < self.config.td3_batch_size:
                    action = np.random.uniform(low, high).astype(np.float32)
                else:
                    action = self.policy.act(state, noise_scale=self.config.td3_exploration_noise)
                next_obs, rewards, terminations, truncations, _ = parallel_env.step({"attacker": action})
                next_state = self.config.flatten_state(next_obs["attacker"], sim_env.max_attackers_sampled)
                reward = rewards["attacker"]
                terminated = terminations["attacker"]
                truncated = truncations["attacker"]
                self.replay.add(state, action, reward, next_state, terminated or truncated)
                state = next_state
                steps_since_update += 1
                if steps_since_update >= max(1, self.config.td3_train_freq_steps):
                    self.policy.train_step(self.replay)
                    steps_since_update = 0
