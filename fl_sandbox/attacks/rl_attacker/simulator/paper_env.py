"""Paper-aligned clipped-median FL simulator for offline TD3 training."""

from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from fl_sandbox.aggregators.rules import AggregationDefender
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.observation import build_paper_clipped_median_observation
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import (
    build_model_from_template,
    capture_weights,
    craft_paper_reversal_update,
)
from fl_sandbox.runtime import Weights


def _fl_num_clients(fl_config, fallback: int = 1) -> int:
    fl = getattr(fl_config, "fl", None)
    return int(getattr(fl, "num_clients", getattr(fl_config, "num_clients", fallback)) or fallback)


def _fl_num_attackers(fl_config, fallback: int = 1) -> int:
    if hasattr(fl_config, "resolved_num_attackers"):
        return int(fl_config.resolved_num_attackers())
    fl = getattr(fl_config, "fl", None)
    return int(getattr(fl, "num_attackers", getattr(fl_config, "num_attackers", fallback)) or fallback)


def _fl_subsample_rate(fl_config, fallback: float = 1.0) -> float:
    fl = getattr(fl_config, "fl", None)
    return float(getattr(fl, "subsample_rate", getattr(fl_config, "subsample_rate", fallback)) or fallback)


def _runtime_value(fl_config, name: str, fallback):
    runtime = getattr(fl_config, "runtime", None)
    return getattr(runtime, name, getattr(fl_config, name, fallback))


def decode_paper_action(action: np.ndarray) -> tuple[float, int]:
    """Decode the original paper's 2D TD3 action into clipped-median parameters."""

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size < 2:
        values = np.pad(values, (0, 2 - values.size))
    values = np.nan_to_num(values[:2], nan=0.0, posinf=1.0, neginf=-1.0)
    values = np.clip(values, -1.0, 1.0)
    gamma = float(values[0] * 14.9 + 15.0)
    local_steps = int(values[1] * 24.0 + 25.0)
    return max(0.1, gamma), max(1, local_steps)


class PaperFLSimulator:
    """Explicit simulator for the paper RL attacker, sourced only from Phase 1 data."""

    def __init__(
        self,
        *,
        model_template,
        distribution,
        defender: AggregationDefender,
        config: RLAttackerConfig,
        fl_config,
        device: torch.device,
        eval_loader=None,
    ) -> None:
        self.model_template = copy.deepcopy(model_template).cpu()
        self.distribution = distribution
        self.defender = defender
        self.config = config
        self.fl_config = fl_config
        self.device = device
        self.eval_loader = eval_loader
        self.current_weights: Weights | None = None
        self.current_loss = 0.0
        self.current_acc = 0.0
        self.round_idx = 0
        self.current_num_attackers = 1

    def reset(self, initial_weights):
        self.current_weights = [layer.copy() for layer in initial_weights]
        self.round_idx = 0
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        self.current_loss, self.current_acc = self._evaluate_metrics(self.current_weights)
        return self._get_state()

    def step(self, action: np.ndarray):
        if self.current_weights is None:
            raise RuntimeError("PaperFLSimulator must be reset before stepping")
        old_weights = [layer.copy() for layer in self.current_weights]
        sampled_clients = max(1, int(_fl_num_clients(self.fl_config) * _fl_subsample_rate(self.fl_config)))
        benign_count = max(0, sampled_clients - self.current_num_attackers)
        benign_weights = self._simulate_benign_updates(old_weights, benign_count)
        malicious = self._simulate_malicious_update(old_weights, action)
        malicious_weights = [[layer.copy() for layer in malicious] for _ in range(self.current_num_attackers)]
        updates = benign_weights + malicious_weights
        self.current_weights = self.defender.aggregate(old_weights, updates, trusted_weights=None) if updates else old_weights
        new_loss, new_acc = self._evaluate_metrics(self.current_weights)
        reward = float(new_loss - self.current_loss)
        self.current_loss = new_loss
        self.current_acc = new_acc
        self.round_idx += 1
        done = self.round_idx >= max(1, int(self.config.simulator_horizon))
        if done and hasattr(self.distribution, "advance_episode"):
            self.distribution.advance_episode()
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        return self._get_state(), reward, done

    def _get_state(self) -> np.ndarray:
        if self.current_weights is None:
            raise RuntimeError("State requested before reset")
        return build_paper_clipped_median_observation(
            self.current_weights,
            num_attackers=self.current_num_attackers,
            tail_layers=self.config.state_tail_layers,
        )

    def _sample_num_attackers(self, *, require_positive: bool = False) -> int:
        total_clients = max(1, _fl_num_clients(self.fl_config))
        total_attackers = min(max(0, _fl_num_attackers(self.fl_config)), total_clients)
        sampled_clients = max(1, int(total_clients * _fl_subsample_rate(self.fl_config)))
        if total_attackers <= 0:
            return 0
        population = [1] * total_attackers + [0] * (total_clients - total_attackers)
        for _ in range(32):
            selected = int(sum(random.sample(population, sampled_clients)))
            if selected > 0 or not require_positive:
                return selected
        return 1 if require_positive else 0

    def _simulate_benign_update(self, old_weights):
        model = build_model_from_template(self.model_template, old_weights, self.device)
        model.train()
        lr = float(_runtime_value(self.fl_config, "lr", 0.05) or 0.05)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        local_epochs = int(getattr(getattr(self.fl_config, "fl", None), "local_epochs", 1) or 1)
        batch_size = int(_runtime_value(self.fl_config, "batch_size", self.config.local_search_batch_size) or 1)
        for _ in range(max(1, local_epochs)):
            images, labels = self.distribution.sample(batch_size, self.device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return capture_weights(model)

    def _simulate_benign_updates(self, old_weights, benign_count: int):
        count = max(0, int(benign_count))
        if count <= 0:
            return []
        parallel_clients = max(1, int(_runtime_value(self.fl_config, "parallel_clients", 1) or 1))
        device_type = str(getattr(getattr(self, "device", None), "type", "cpu"))
        if device_type == "cuda":
            parallel_clients = 1
        workers = min(parallel_clients, count)
        if workers <= 1:
            return [self._simulate_benign_update(old_weights) for _ in range(count)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda _: self._simulate_benign_update(old_weights), range(count)))

    def _simulate_malicious_update(self, old_weights, action):
        gamma, local_steps = decode_paper_action(action)
        return craft_paper_malicious_update(
            model_template=self.model_template,
            old_weights=old_weights,
            distribution=self.distribution,
            device=self.device,
            lr=float(self.config.paper_local_lr),
            local_steps=local_steps,
            gamma=gamma,
            batch_size=int(self.config.local_search_batch_size),
        )

    def _evaluate_metrics(self, weights) -> tuple[float, float]:
        if self.eval_loader is not None:
            return self._evaluate_loader_metrics(weights)
        return self._evaluate_distribution_metrics(weights)

    def _evaluate_distribution_metrics(self, weights) -> tuple[float, float]:
        model = build_model_from_template(self.model_template, weights, self.device)
        model.eval()
        images, labels = self.distribution.sample(self.config.local_search_batch_size, self.device)
        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()
        return float(loss.item()), float(acc.item())

    def _evaluate_loader_metrics(self, weights) -> tuple[float, float]:
        model = build_model_from_template(self.model_template, weights, self.device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for images, labels in self.eval_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                total_loss += float(loss.item())
                total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
                total_count += int(labels.numel())
        if total_count <= 0:
            return self._evaluate_distribution_metrics(weights)
        return total_loss / total_count, total_correct / total_count


def craft_paper_malicious_update(
    *,
    model_template,
    old_weights: Weights,
    distribution,
    device: torch.device,
    lr: float,
    local_steps: int,
    gamma: float,
    batch_size: int,
) -> Weights:
    model = build_model_from_template(model_template, old_weights, device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=max(1e-6, float(lr)))
    for _ in range(max(1, int(local_steps))):
        images, labels = distribution.sample(batch_size, device)
        loss = F.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.data.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
    return craft_paper_reversal_update(old_weights, capture_weights(model), gamma_scale=gamma)


class PaperAttackerPolicyGymEnv:
    """Gymnasium wrapper around the paper simulator."""

    metadata = {"render_modes": []}

    def __init__(self, simulator: PaperFLSimulator, rl_config: RLAttackerConfig, defense_type: str, initial_weights) -> None:
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
        del options
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
