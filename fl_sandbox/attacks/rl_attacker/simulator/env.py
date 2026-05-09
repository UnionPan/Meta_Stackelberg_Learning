"""Gymnasium simulator for Tianshou-backed RL attacker training."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.observation import build_observation_from_state
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import (
    build_model_from_template,
    capture_weights,
    craft_malicious_update,
    match_update_norm,
    update_norm,
)
from fl_sandbox.attacks.rl_attacker.simulator.reward import DefaultRewardFn, RewardInputs


class SimulatedFLEnv:
    """Small FL world model used as the policy-training environment."""

    def __init__(self, *, model_template, proxy_buffer, defender, config: RLAttackerConfig, fl_config, device: torch.device) -> None:
        self.model_template = copy.deepcopy(model_template).cpu()
        self.proxy_buffer = proxy_buffer
        self.defender = defender
        self.config = config
        self.fl_config = fl_config
        self.device = device
        self.reward_fn = DefaultRewardFn(config)
        self.current_weights = None
        self.current_loss = 0.0
        self.current_acc = 0.0
        self.round_idx = 0
        self.current_num_attackers = 0
        self.max_attackers_sampled = 1
        self.previous_action = np.zeros(config.action_dim(defender.defense_type), dtype=np.float32)
        self.previous_bypass_score = 0.0

    def reset(self, initial_weights):
        self.current_weights = [layer.copy() for layer in initial_weights]
        self.round_idx = 0
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        self.max_attackers_sampled = max(1, self.current_num_attackers)
        self.current_loss, self.current_acc = self._evaluate_proxy_metrics(self.current_weights)
        self.previous_action = np.zeros(self.config.action_dim(self.defender.defense_type), dtype=np.float32)
        self.previous_bypass_score = 0.0
        return self._get_state()

    def step(self, action: np.ndarray):
        if self.current_weights is None:
            raise RuntimeError("SimulatedFLEnv must be reset before stepping")
        action = np.asarray(action, dtype=np.float32)
        benign_count = max(0, max(1, int(self.fl_config.num_clients * self.fl_config.subsample_rate)) - self.current_num_attackers)
        benign_weights = [self._simulate_benign_update(self.current_weights) for _ in range(benign_count)]
        malicious_weights = [self._simulate_malicious_weight(self.current_weights, action) for _ in range(self.current_num_attackers)]
        benign_norm_mean = float(np.mean([update_norm(self.current_weights, w) for w in benign_weights])) if benign_weights else 0.0
        malicious_norm_mean = float(np.mean([update_norm(self.current_weights, w) for w in malicious_weights])) if malicious_weights else 0.0
        updates = benign_weights + malicious_weights
        if updates:
            self.current_weights = self.defender.aggregate(self.current_weights, updates, trusted_weights=None)
        new_loss, new_acc = self._evaluate_proxy_metrics(self.current_weights)
        bypass = 1.0 if malicious_weights else 0.0
        norm_penalty = 0.0
        if benign_norm_mean > 0 and malicious_norm_mean > 0:
            target = benign_norm_mean * self.config.target_norm_ratio(self.defender.defense_type)
            norm_penalty = max(0.0, malicious_norm_mean - target) / benign_norm_mean
        reward = self.reward_fn(
            RewardInputs(
                loss_delta=new_loss - self.current_loss,
                acc_delta=self.current_acc - new_acc,
                bypass_score=bypass,
                action=action,
                previous_action=self.previous_action,
                norm_penalty=norm_penalty,
            )
        )
        self.previous_action = action.copy()
        self.previous_bypass_score = bypass
        self.current_loss = new_loss
        self.current_acc = new_acc
        self.round_idx += 1
        done = self.round_idx >= max(1, self.config.simulator_horizon)
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        self.max_attackers_sampled = max(self.max_attackers_sampled, self.current_num_attackers)
        return self._get_state(), reward, done

    def _sample_num_attackers(self, *, require_positive: bool = False) -> int:
        total_clients = max(1, int(getattr(self.fl_config, "num_clients", 1) or 1))
        configured_attackers = getattr(self.fl_config, "num_attackers", 1)
        total_attackers = min(max(0, int(configured_attackers or 0)), total_clients)
        subsample_rate = float(getattr(self.fl_config, "subsample_rate", 1.0) or 1.0)
        sampled_clients = max(1, int(total_clients * subsample_rate))
        if total_attackers <= 0:
            return 0
        population = [1] * total_attackers + [0] * (total_clients - total_attackers)
        for _ in range(32):
            selected_attackers = int(sum(random.sample(population, sampled_clients)))
            if selected_attackers > 0 or not require_positive:
                return selected_attackers
        return 1 if require_positive else 0

    def _get_state_dict(self) -> dict[str, Any]:
        if self.current_weights is None:
            raise RuntimeError("State requested before reset")
        model = build_model_from_template(self.model_template, self.current_weights, self.device)
        from fl_sandbox.models import get_compressed_state
        compressed, _ = get_compressed_state(model, num_tail_layers=self.config.state_tail_layers)
        return {"pram": compressed.astype(np.float32), "num_attacker": int(self.current_num_attackers)}

    def _get_state(self) -> np.ndarray:
        return build_observation_from_state(self._get_state_dict(), self.max_attackers_sampled)

    def _simulate_benign_update(self, old_weights):
        model = build_model_from_template(self.model_template, old_weights, self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(getattr(self.fl_config, "lr", 0.05) or 0.05))
        for _ in range(max(1, int(getattr(self.fl_config, "local_epochs", 1) or 1))):
            batch_size = int(getattr(self.fl_config, "batch_size", self.config.local_search_batch_size) or 1)
            images, labels = self.proxy_buffer.sample(batch_size, self.device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return capture_weights(model)

    def _simulate_malicious_weight(self, old_weights, action):
        params = decode_action(action, self.defender.defense_type, self.config)
        crafted = craft_malicious_update(
            model_template=self.model_template,
            old_weights=old_weights,
            proxy_buffer=self.proxy_buffer,
            device=self.device,
            params=params,
            search_batch_size=self.config.local_search_batch_size,
            state_tail_layers=self.config.state_tail_layers,
        )
        target_norm = self._sampled_target_norm(old_weights)
        return match_update_norm(old_weights, crafted, target_norm=target_norm)

    def _evaluate_proxy_metrics(self, weights) -> tuple[float, float]:
        model = build_model_from_template(self.model_template, weights, self.device)
        images, labels = self.proxy_buffer.sample(self.config.local_search_batch_size, self.device)
        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = float((torch.argmax(logits, dim=1) == labels).float().mean().item())
        return float(loss.item()), acc

    def _sampled_target_norm(self, old_weights) -> float:
        benign_samples = [self._simulate_benign_update(old_weights) for _ in range(max(2, self.current_num_attackers or 1))]
        mean = float(np.mean([update_norm(old_weights, w) for w in benign_samples])) if benign_samples else 0.0
        return mean * self.config.target_norm_ratio(self.defender.defense_type)


class AttackerPolicyGymEnv:
    """Gymnasium-compatible wrapper around `SimulatedFLEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, simulator: SimulatedFLEnv, rl_config: RLAttackerConfig, defense_type: str, initial_weights) -> None:
        import gymnasium as gym

        self.simulator = simulator
        self.rl_config = rl_config
        self.defense_type = defense_type
        self.initial_weights = [layer.copy() for layer in initial_weights]
        sample_obs = np.asarray(self.simulator.reset(self.initial_weights), dtype=np.float32)
        low, high = rl_config.action_bounds(defense_type)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        obs = np.asarray(self.simulator.reset(self.initial_weights), dtype=np.float32)
        return obs, {"defense_type": self.defense_type}

    def step(self, action):
        obs, reward, done = self.simulator.step(np.asarray(action, dtype=np.float32))
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, {}

    def render(self):
        return None

    def close(self) -> None:
        return None


AttackerRLEnv = AttackerPolicyGymEnv
