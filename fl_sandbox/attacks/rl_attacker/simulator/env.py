"""Gymnasium simulator for Tianshou-backed RL attacker training."""

from __future__ import annotations

import copy
import inspect
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.krum_projection import (
    fast_krum_geometry_metrics,
    fast_legacy_krum_surrogate_update,
    krum_geometry_from_updates,
    simulate_krum_benign_surrogates,
)
from fl_sandbox.attacks.rl_attacker.observation import (
    ProjectedObservationBuilder,
    build_legacy_clipped_median_observation,
    build_legacy_scaleaware_observation,
)
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import (
    build_model_from_template,
    capture_weights,
    craft_malicious_update,
    match_update_norm,
    update_norm,
)
from fl_sandbox.attacks.rl_attacker.simulator.reward import DefaultRewardFn, RewardInputs


class _StrictReproductionDataLoaderSource:
    """DataLoader-backed attacker data source matching the legacy growth schedule."""

    uses_dataloader = True

    def __init__(self, proxy_buffer, sample_limit: int) -> None:
        self.proxy_buffer = proxy_buffer
        self.sample_limit = max(0, int(sample_limit))
        self._dataset = self._build_dataset()
        self._iterators: dict[int, object] = {}

    def __len__(self) -> int:
        return 0 if self._dataset is None else len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    def _build_dataset(self):
        samples = list(self.proxy_buffer._samples)[: self.sample_limit]
        if not samples:
            return None
        images, labels = zip(*samples)
        return TensorDataset(torch.stack(images), torch.stack(labels).long())

    def _loader(self, batch_size: int) -> DataLoader:
        batch_size = min(max(1, int(batch_size)), max(1, len(self)))
        return DataLoader(self._dataset, batch_size=batch_size, shuffle=True)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dataset is None or len(self._dataset) == 0:
            return self.proxy_buffer.sample(batch_size, device)
        batch_size = min(max(1, int(batch_size)), len(self._dataset))
        iterator = self._iterators.get(batch_size)
        if iterator is None:
            iterator = iter(self._loader(batch_size))
            self._iterators[batch_size] = iterator
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(self._loader(batch_size))
            self._iterators[batch_size] = iterator
            images, labels = next(iterator)
        return images.to(device), labels.to(device)


class SimulatedFLEnv:
    """Small FL world model used as the policy-training environment."""

    def __init__(
        self,
        *,
        model_template,
        proxy_buffer,
        defender,
        config: RLAttackerConfig,
        fl_config,
        device: torch.device,
        eval_loader=None,
    ) -> None:
        self.model_template = copy.deepcopy(model_template).cpu()
        self.proxy_buffer = proxy_buffer
        self.eval_loader = eval_loader
        self.defender = defender
        self.config = config
        self.config.validate_defense(defender.defense_type)
        self.fl_config = fl_config
        self.device = device
        self.reward_fn = DefaultRewardFn(config)
        self.current_weights = None
        self.previous_weights = None
        self.fast_metric_origin_weights = None
        self.last_aggregate_update = None
        self.current_loss = 0.0
        self.current_acc = 0.0
        self.round_idx = 0
        self.current_num_attackers = 0
        self.max_attackers_sampled = 1
        self.strict_reproduction_epoch = 0
        self.strict_reproduction_sample_limit = len(proxy_buffer)
        self.strict_reproduction_data_source = None
        self.previous_action = np.zeros(config.action_dim(defender.defense_type), dtype=np.float32)
        self.previous_bypass_score = 0.0
        self.observation_builder = ProjectedObservationBuilder(
            config=config,
            action_dim=config.action_dim(defender.defense_type),
        )

    def reset(self, initial_weights):
        self.current_weights = [layer.copy() for layer in initial_weights]
        self.previous_weights = [layer.copy() for layer in initial_weights]
        self.fast_metric_origin_weights = [layer.copy() for layer in initial_weights]
        self.last_aggregate_update = [np.zeros_like(layer) for layer in initial_weights]
        self.round_idx = 0
        self._advance_strict_reproduction_schedule()
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        self.max_attackers_sampled = max(1, self.current_num_attackers)
        self.current_loss, self.current_acc = self._evaluate_metrics(self.current_weights)
        self.previous_action = np.zeros(self.config.action_dim(self.defender.defense_type), dtype=np.float32)
        self.previous_bypass_score = 0.0
        self.observation_builder.history.clear()
        return self._get_state()

    def step(self, action: np.ndarray):
        if self.current_weights is None:
            raise RuntimeError("SimulatedFLEnv must be reset before stepping")
        action = np.asarray(action, dtype=np.float32)
        benign_count = max(0, max(1, int(self.fl_config.num_clients * self.fl_config.subsample_rate)) - self.current_num_attackers)
        old_weights = [layer.copy() for layer in self.current_weights]
        benign_weights = self._simulate_benign_weights(self.current_weights, benign_count)
        if self.config.uses_legacy_reversal_attack() and self.current_num_attackers > 0:
            malicious = self._call_simulate_malicious_weight(self.current_weights, action, benign_weights=benign_weights)
            malicious_weights = [
                [layer.copy() for layer in malicious]
                for _ in range(self.current_num_attackers)
            ]
        else:
            malicious_weights = [
                self._call_simulate_malicious_weight(self.current_weights, action, benign_weights=benign_weights)
                for _ in range(self.current_num_attackers)
            ]
        benign_norm_mean = float(np.mean([update_norm(self.current_weights, w) for w in benign_weights])) if benign_weights else 0.0
        malicious_norm_mean = float(np.mean([update_norm(self.current_weights, w) for w in malicious_weights])) if malicious_weights else 0.0
        updates = benign_weights + malicious_weights
        if updates:
            self.current_weights = self._aggregate_simulated_updates(
                self.current_weights,
                benign_weights,
                malicious_weights,
            )
        self.previous_weights = old_weights
        self.last_aggregate_update = [new - old for old, new in zip(old_weights, self.current_weights)]
        new_loss, new_acc = self._evaluate_metrics(self.current_weights)
        bypass = self._simulated_bypass_score(old_weights, benign_weights, malicious_weights)
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

    def _get_state(self) -> np.ndarray:
        if self.current_weights is None:
            raise RuntimeError("State requested before reset")
        if self.config.uses_legacy_reversal_attack():
            if self.config.uses_scaleaware_legacy_observation():
                return build_legacy_scaleaware_observation(
                    self.current_weights,
                    num_attackers=self.current_num_attackers,
                    round_idx=self.round_idx,
                    total_rounds=max(1, self.config.simulator_horizon),
                )
            return build_legacy_clipped_median_observation(
                self.current_weights,
                num_attackers=self.current_num_attackers,
            )
        return self.observation_builder.build(
            weights=self.current_weights,
            previous_weights=self.previous_weights,
            last_aggregate_update=self.last_aggregate_update,
            last_action=self.previous_action,
            last_bypass_score=self.previous_bypass_score,
            round_idx=self.round_idx,
            total_rounds=max(1, self.config.simulator_horizon),
            defense_type=self.defender.defense_type,
        )

    def _simulate_benign_update(self, old_weights):
        model = build_model_from_template(self.model_template, old_weights, self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(getattr(self.fl_config, "lr", 0.05) or 0.05))
        proxy_buffer = self.active_proxy_buffer()
        for _ in range(max(1, int(getattr(self.fl_config, "local_epochs", 1) or 1))):
            batch_size = int(getattr(self.fl_config, "batch_size", self.config.local_search_batch_size) or 1)
            images, labels = proxy_buffer.sample(batch_size, self.device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return capture_weights(model)

    def _simulate_benign_weights(self, old_weights, benign_count: int):
        benign_count = max(0, int(benign_count))
        if benign_count <= 0:
            return []
        if not self.config.uses_legacy_krum_geometry():
            return [self._simulate_benign_update(old_weights) for _ in range(benign_count)]
        return simulate_krum_benign_surrogates(
            old_weights=old_weights,
            last_aggregate_update=self.last_aggregate_update,
            benign_count=benign_count,
            seed=int(getattr(self.config, "seed", 0)),
            round_idx=int(self.round_idx),
        )

    def _simulate_malicious_weight(self, old_weights, action, *, benign_weights):
        params = decode_action(action, self.defender.defense_type, self.config)
        if self.config.uses_legacy_krum_geometry() and str(self.defender.defense_type).lower() in {"krum", "multi_krum"}:
            return fast_legacy_krum_surrogate_update(
                old_weights=old_weights,
                benign_weights=benign_weights,
                gamma_scale=float(params.gamma_scale),
                local_steps=int(params.local_steps),
                steps_center=float(self.config.legacy_krum_steps_center),
                current_num_attackers=int(self.current_num_attackers),
                seed=int(getattr(self.config, "seed", 0)),
                round_idx=int(self.round_idx),
            )
        crafted = craft_malicious_update(
            model_template=self.model_template,
            old_weights=old_weights,
            proxy_buffer=self.active_proxy_buffer(),
            device=self.device,
            params=params,
            search_batch_size=self.config.local_search_batch_size,
            state_tail_layers=self.config.state_tail_layers,
        )
        if self.config.uses_legacy_reversal_attack():
            return crafted
        target_norm = self._sampled_target_norm(old_weights)
        return match_update_norm(old_weights, crafted, target_norm=target_norm)

    def _call_simulate_malicious_weight(self, old_weights, action, *, benign_weights):
        signature = inspect.signature(self._simulate_malicious_weight)
        if "benign_weights" in signature.parameters:
            return self._simulate_malicious_weight(old_weights, action, benign_weights=benign_weights)
        return self._simulate_malicious_weight(old_weights, action)

    def _simulated_bypass_score(self, old_weights, benign_weights, malicious_weights) -> float:
        if not malicious_weights:
            return 0.0
        defense = str(self.defender.defense_type).lower()
        if self.config.uses_legacy_krum_geometry() and defense in {"krum", "multi_krum"}:
            return 1.0
        if defense not in {"krum", "multi_krum"}:
            return 1.0
        old_vec = np.concatenate([np.asarray(layer, dtype=np.float32).reshape(-1) for layer in old_weights])
        all_weights = list(benign_weights) + list(malicious_weights)
        update_vectors = np.stack(
            [
                np.concatenate([np.asarray(layer, dtype=np.float32).reshape(-1) for layer in weights]) - old_vec
                for weights in all_weights
            ],
            axis=0,
        )
        malicious_indices = list(range(len(benign_weights), len(all_weights)))
        stats = krum_geometry_from_updates(
            update_vectors,
            malicious_indices,
            num_byzantine=int(getattr(self.defender, "krum_attackers", len(malicious_weights))),
        )
        return float(stats.selected)

    def _aggregate_simulated_updates(self, old_weights, benign_weights, malicious_weights):
        if self.config.uses_legacy_krum_geometry() and malicious_weights:
            return [layer.copy() for layer in malicious_weights[0]]
        return self.defender.aggregate(old_weights, list(benign_weights) + list(malicious_weights), trusted_weights=None)

    def active_proxy_buffer(self):
        if not self.config.uses_strict_reproduction():
            return self.proxy_buffer
        if self.strict_reproduction_data_source is None:
            self.strict_reproduction_data_source = _StrictReproductionDataLoaderSource(
                self.proxy_buffer,
                self.strict_reproduction_sample_limit,
            )
        return self.strict_reproduction_data_source

    def _advance_strict_reproduction_schedule(self) -> None:
        if not self.config.uses_strict_reproduction():
            self.strict_reproduction_sample_limit = len(self.proxy_buffer)
            return
        self.strict_reproduction_epoch += 1
        self.strict_reproduction_sample_limit = self.config.strict_reproduction_sample_limit(
            epoch=self.strict_reproduction_epoch,
            buffer_size=len(self.proxy_buffer),
        )
        self.strict_reproduction_data_source = _StrictReproductionDataLoaderSource(
            self.proxy_buffer,
            self.strict_reproduction_sample_limit,
        )

    def _evaluate_metrics(self, weights) -> tuple[float, float]:
        if self.config.uses_legacy_krum_geometry():
            return self._evaluate_fast_geometry_metrics(weights)
        if self.config.uses_strict_reproduction() and self.eval_loader is not None:
            return self._evaluate_eval_metrics(weights)
        return self._evaluate_proxy_metrics(weights)

    def _evaluate_fast_geometry_metrics(self, weights) -> tuple[float, float]:
        origin = self.fast_metric_origin_weights or self.previous_weights or weights
        return fast_krum_geometry_metrics(origin_weights=origin, weights=weights)

    def _evaluate_proxy_metrics(self, weights) -> tuple[float, float]:
        model = build_model_from_template(self.model_template, weights, self.device)
        images, labels = self.active_proxy_buffer().sample(self.config.local_search_batch_size, self.device)
        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = float((torch.argmax(logits, dim=1) == labels).float().mean().item())
        return float(loss.item()), acc

    def _evaluate_eval_metrics(self, weights) -> tuple[float, float]:
        model = build_model_from_template(self.model_template, weights, self.device)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        model.eval()
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
            return self._evaluate_proxy_metrics(weights)
        return total_loss / total_count, total_correct / total_count

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
