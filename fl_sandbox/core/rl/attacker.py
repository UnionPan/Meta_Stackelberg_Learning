"""Paper-inspired RL attacker for federated learning sandbox experiments.

This module implements a practical sandbox adaptation of the NeurIPS 2022
RL-attacker:

1. learn a proxy benign-data distribution from observed server updates;
2. build a simulated FL environment with that proxy distribution;
3. train a continuous attacker policy with TD3; and
4. execute the learned policy online to craft malicious model updates.

The implementation intentionally keeps the public API small so it can slot into
the existing `SandboxAttack` interface without rewriting the sandbox runner.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Deque, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.cnn import get_compressed_state

from fl_sandbox.core.runtime import RoundContext, Weights

from ..defender import AggregationDefender
from ..attacks.poisoning import craft_ipm
from .pz_env import AttackerPolicyParallelEnv


def _tail_named_parameters(model: nn.Module, num_tail_layers: int) -> list[tuple[str, nn.Parameter]]:
    params = list(model.named_parameters())
    return params[-max(1, num_tail_layers) :]


def _capture_weights(model: nn.Module) -> Weights:
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def _load_weights(model: nn.Module, weights: Weights, device: torch.device) -> None:
    with torch.no_grad():
        for target, source in zip(model.state_dict().values(), weights):
            tensor = torch.as_tensor(source, device=device, dtype=target.dtype)
            target.copy_(tensor.reshape_as(target))


def _build_model_from_template(template: nn.Module, weights: Weights, device: torch.device) -> nn.Module:
    model = copy.deepcopy(template).to(device)
    _load_weights(model, weights, device)
    return model


def _weights_to_vector(weights: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(layer).ravel() for layer in weights], axis=0)


def _vectorize_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def _tv_loss(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        return torch.zeros((), device=images.device, dtype=images.dtype)
    loss_h = torch.mean(torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]))
    return loss_h + loss_w


class ProxyDatasetBuffer:
    """A small replay-style buffer for proxy benign data samples."""

    def __init__(self, max_samples: int) -> None:
        self.max_samples = max(1, int(max_samples))
        self.images: Deque[torch.Tensor] = deque(maxlen=self.max_samples)
        self.labels: Deque[torch.Tensor] = deque(maxlen=self.max_samples)
        self.input_min = -1.0
        self.input_max = 1.0
        self.image_shape: Optional[torch.Size] = None
        self.num_classes: Optional[int] = None

    def __len__(self) -> int:
        return len(self.images)

    def extend(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.numel() == 0 or labels.numel() == 0:
            return
        images_cpu = images.detach().cpu()
        labels_cpu = labels.detach().cpu().long()
        if self.image_shape is None:
            self.image_shape = images_cpu[0].shape
        label_max = int(labels_cpu.max().item()) if labels_cpu.numel() > 0 else 0
        self.num_classes = max(self.num_classes or 0, label_max + 1)
        self.input_min = min(self.input_min, float(images_cpu.min().item()))
        self.input_max = max(self.input_max, float(images_cpu.max().item()))
        for image, label in zip(images_cpu, labels_cpu):
            self.images.append(image.clone())
            self.labels.append(label.clone())

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.images:
            raise RuntimeError("ProxyDatasetBuffer is empty")
        take = min(max(1, int(batch_size)), len(self.images))
        indices = np.random.choice(len(self.images), size=take, replace=len(self.images) < take)
        batch_x = torch.stack([list(self.images)[idx] for idx in indices]).to(device)
        batch_y = torch.stack([list(self.labels)[idx] for idx in indices]).to(device)
        return batch_x, batch_y


class ConvDenoiser(nn.Module):
    """A tiny convolutional denoiser for reconstructed proxy samples."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class GradientDistributionLearner:
    """Maintain a proxy benign-data distribution reconstructed from server updates."""

    def __init__(self, config: "RLAttackerConfig") -> None:
        self.config = config
        self.buffer = ProxyDatasetBuffer(max_samples=config.proxy_buffer_limit)
        self.device = torch.device("cpu")
        self.initialized = False
        self.denoiser: Optional[ConvDenoiser] = None
        self.denoiser_optimizer: Optional[torch.optim.Optimizer] = None
        self.last_observed_grad: Optional[np.ndarray] = None

    def initialize_from_loader(self, loader: Optional[DataLoader], device: torch.device) -> None:
        if self.initialized or loader is None:
            return
        self.device = device
        images_accum = []
        labels_accum = []
        target = max(1, self.config.seed_samples)
        for images, labels in loader:
            images_accum.append(images.detach().cpu())
            labels_accum.append(labels.detach().cpu())
            if sum(batch.shape[0] for batch in images_accum) >= target:
                break
        if not images_accum:
            return
        images = torch.cat(images_accum, dim=0)[:target]
        labels = torch.cat(labels_accum, dim=0)[:target]
        self.buffer.extend(images, labels)
        self._build_denoiser()
        self._train_denoiser()
        self.initialized = True

    def observe_server_update(
        self,
        *,
        previous_weights: Optional[Weights],
        current_weights: Weights,
        model_template: nn.Module,
        server_lr: float,
        round_gap: int,
    ) -> None:
        if not self.initialized or previous_weights is None or len(self.buffer) == 0:
            return
        gap = max(1, int(round_gap))
        grad_vec = (_weights_to_vector(previous_weights) - _weights_to_vector(current_weights)) / max(server_lr * gap, 1e-8)
        self.last_observed_grad = grad_vec.astype(np.float32)
        reconstructed_x, reconstructed_y = self._invert_gradient(current_weights, grad_vec, model_template)
        if reconstructed_x is None or reconstructed_y is None:
            return
        if self.denoiser is not None:
            with torch.no_grad():
                reconstructed_x = self.denoiser(reconstructed_x.to(self.device)).cpu()
        reconstructed_x = reconstructed_x.clamp(self.buffer.input_min, self.buffer.input_max)
        self.buffer.extend(reconstructed_x, reconstructed_y)
        self._train_denoiser()

    def _build_denoiser(self) -> None:
        if self.buffer.image_shape is None or self.denoiser is not None:
            return
        channels = int(self.buffer.image_shape[0])
        self.denoiser = ConvDenoiser(channels).to(self.device)
        self.denoiser_optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=1e-3)

    def _train_denoiser(self) -> None:
        if self.denoiser is None or self.denoiser_optimizer is None or len(self.buffer) == 0:
            return
        self.denoiser.train()
        batch_size = min(self.config.local_search_batch_size, len(self.buffer))
        for _ in range(max(1, self.config.denoiser_epochs)):
            clean_x, _ = self.buffer.sample(batch_size, self.device)
            noise = self.config.denoiser_noise_std * torch.randn_like(clean_x)
            noisy_x = (clean_x + noise).clamp(self.buffer.input_min, self.buffer.input_max)
            recon = self.denoiser(noisy_x)
            loss = F.mse_loss(recon, clean_x)
            self.denoiser_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.denoiser_optimizer.step()

    def _invert_gradient(
        self,
        current_weights: Weights,
        grad_vec: np.ndarray,
        model_template: nn.Module,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.buffer.image_shape is None or self.buffer.num_classes is None:
            return None, None
        model = _build_model_from_template(model_template, current_weights, self.device)
        model.eval()
        batch_size = max(1, self.config.reconstruction_batch_size)
        dummy_x = torch.empty((batch_size, *self.buffer.image_shape), device=self.device).uniform_(
            self.buffer.input_min,
            self.buffer.input_max,
        )
        dummy_x.requires_grad_(True)
        dummy_logits = torch.zeros((batch_size, self.buffer.num_classes), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([dummy_x, dummy_logits], lr=self.config.inversion_lr)
        target_grad = torch.as_tensor(grad_vec, device=self.device, dtype=torch.float32)
        tail_params = [param for _, param in _tail_named_parameters(model, self.config.state_tail_layers)]
        for _ in range(max(1, self.config.inversion_steps)):
            log_probs = F.log_softmax(model(dummy_x), dim=1)
            probs = F.softmax(dummy_logits, dim=1)
            dummy_loss = -(probs * log_probs).sum(dim=1).mean()
            grads = torch.autograd.grad(dummy_loss, tail_params, create_graph=True)
            dummy_grad_vec = _vectorize_tensors(grads)
            loss = 1.0 - F.cosine_similarity(dummy_grad_vec.unsqueeze(0), target_grad[: dummy_grad_vec.numel()].unsqueeze(0)).mean()
            loss = loss + self.config.tv_weight * _tv_loss(dummy_x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                dummy_x.clamp_(self.buffer.input_min, self.buffer.input_max)
        labels = torch.argmax(dummy_logits.detach(), dim=1).cpu()
        return dummy_x.detach().cpu(), labels


class ReplayBuffer:
    """Simple replay buffer for TD3."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = max(1, int(capacity))
        self.storage: Deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.storage.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch_size = min(max(1, int(batch_size)), len(self.storage))
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(list(self.storage)[idx] for idx in indices))
        return (
            torch.as_tensor(np.stack(states), device=device, dtype=torch.float32),
            torch.as_tensor(np.stack(actions), device=device, dtype=torch.float32),
            torch.as_tensor(np.asarray(rewards)[:, None], device=device, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), device=device, dtype=torch.float32),
            torch.as_tensor(np.asarray(dones)[:, None], device=device, dtype=torch.float32),
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_low: np.ndarray, action_high: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_low.shape[0]),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        norm_action = self.net(state)
        return self.action_low + 0.5 * (norm_action + 1.0) * (self.action_high - self.action_low)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state, action], dim=1)
        return self.q1(inputs), self.q2(inputs)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=1))


class TD3Agent:
    """A lightweight TD3 implementation tailored to the attacker simulator."""

    def __init__(self, state_dim: int, action_low: np.ndarray, action_high: np.ndarray, config: "RLAttackerConfig", device: torch.device) -> None:
        self.config = config
        self.device = device
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.actor = Actor(state_dim, self.action_low, self.action_high).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, self.action_low.shape[0]).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.policy_lr)
        self.total_updates = 0

    def act(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        state_tensor = torch.as_tensor(state[None, :], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if noise_scale > 0:
            action = action + np.random.normal(0.0, noise_scale, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_low, self.action_high)

    def train_step(self, replay: ReplayBuffer) -> None:
        if len(replay) < self.config.td3_batch_size:
            return
        states, actions, rewards, next_states, dones = replay.sample(self.config.td3_batch_size, self.device)
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.td3_policy_noise
            noise = noise.clamp(-self.config.td3_noise_clip, self.config.td3_noise_clip)
            next_actions = self.actor_target(next_states) + noise
            low = torch.as_tensor(self.action_low, device=self.device)
            high = torch.as_tensor(self.action_high, device=self.device)
            next_actions = torch.max(torch.min(next_actions, high), low)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.config.td3_gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_updates += 1
        if self.total_updates % self.config.td3_policy_delay != 0:
            return

        actor_loss = -self.critic.q1_value(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.config.td3_tau).add_(self.config.td3_tau * source_param.data)


class SimulatedFLEnv:
    """Approximate FL environment built from the reconstructed proxy distribution."""

    def __init__(
        self,
        *,
        model_template: nn.Module,
        proxy_buffer: ProxyDatasetBuffer,
        defender: AggregationDefender,
        config: "RLAttackerConfig",
        fl_config,
        device: torch.device,
    ) -> None:
        self.model_template = copy.deepcopy(model_template).cpu()
        self.proxy_buffer = proxy_buffer
        self.defender = defender
        self.config = config
        self.fl_config = fl_config
        self.device = device
        self.round_idx = 0
        self.current_weights: Optional[Weights] = None
        self.max_attackers_sampled = max(1, min(fl_config.num_attackers, max(1, int(fl_config.num_clients * fl_config.subsample_rate))))
        self.current_num_attackers = 0
        self.current_loss = 0.0

    def reset(self, initial_weights: Weights) -> np.ndarray:
        self.round_idx = 0
        self.current_weights = [layer.copy() for layer in initial_weights]
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        self.current_loss = self._evaluate_proxy_loss(self.current_weights)
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if self.current_weights is None:
            raise RuntimeError("SimulatedFLEnv must be reset before stepping")
        benign_count = max(0, max(1, int(self.fl_config.num_clients * self.fl_config.subsample_rate)) - self.current_num_attackers)
        benign_weights = [self._simulate_benign_update(self.current_weights) for _ in range(benign_count)]
        malicious_weights = [
            self._simulate_malicious_weight(self.current_weights, action)
            for _ in range(self.current_num_attackers)
        ]
        trusted_weights = self._trusted_reference_update(self.current_weights)
        updates = benign_weights + malicious_weights
        if updates:
            self.current_weights = self.defender.aggregate(self.current_weights, updates, trusted_weights=trusted_weights)
        new_loss = self._evaluate_proxy_loss(self.current_weights)
        reward = new_loss - self.current_loss
        self.current_loss = new_loss
        self.round_idx += 1
        done = self.round_idx >= max(1, self.config.simulator_horizon)
        self.current_num_attackers = self._sample_num_attackers(require_positive=True)
        return self._get_state(), reward, done

    def _sample_num_attackers(self, *, require_positive: bool = False) -> int:
        total_clients = max(1, int(self.fl_config.num_clients))
        total_attackers = min(max(0, int(self.fl_config.num_attackers)), total_clients)
        sampled_clients = max(1, int(total_clients * float(self.fl_config.subsample_rate)))
        if total_attackers <= 0:
            return 0
        population = [1] * total_attackers + [0] * (total_clients - total_attackers)
        for _ in range(32):
            sampled = random.sample(population, sampled_clients)
            selected_attackers = int(sum(sampled))
            if selected_attackers > 0 or not require_positive:
                return selected_attackers
        return 1 if require_positive else 0

    def _get_state_dict(self) -> dict[str, Any]:
        if self.current_weights is None:
            raise RuntimeError("State requested before reset")
        model = _build_model_from_template(self.model_template, self.current_weights, self.device)
        return self.config.format_state(model, self.current_num_attackers, self.max_attackers_sampled)

    def _get_state(self) -> np.ndarray:
        return self.config.flatten_state(self._get_state_dict(), self.max_attackers_sampled)

    def _simulate_benign_update(self, old_weights: Weights) -> Weights:
        model = _build_model_from_template(self.model_template, old_weights, self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fl_config.lr)
        for _ in range(max(1, self.fl_config.local_epochs)):
            images, labels = self.proxy_buffer.sample(self.fl_config.batch_size, self.device)
            loss = F.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return _capture_weights(model)

    def _trusted_reference_update(self, old_weights: Weights) -> Optional[Weights]:
        if self.defender.defense_type != "fltrust":
            return None
        model = _build_model_from_template(self.model_template, old_weights, self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fl_config.lr)
        images, labels = self.proxy_buffer.sample(max(1, self.fl_config.fltrust_root_size), self.device)
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return _capture_weights(model)

    def _simulate_malicious_weight(self, old_weights: Weights, action: np.ndarray) -> Weights:
        action_params = self.config.decode_action(action, self.defender.defense_type)
        if self.defender.defense_type == "fltrust":
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
        return craft_ipm(old_weights, trained_weights, scale=action_params.gamma_scale)

    def _evaluate_proxy_loss(self, weights: Weights) -> float:
        model = _build_model_from_template(self.model_template, weights, self.device)
        images, labels = self.proxy_buffer.sample(self.config.local_search_batch_size, self.device)
        with torch.no_grad():
            loss = F.cross_entropy(model(images), labels)
        return float(loss.item())


@dataclass
class DecodedAction:
    gamma_scale: float
    local_steps: int
    lambda_stealth: float
    local_search_lr: float


@dataclass
class RLAttackerConfig:
    """Config for the paper-inspired RL attacker.

    The defaults are sandbox-scaled to keep the attacker usable in this repo.
    Larger values can be supplied from the CLI to move closer to the paper
    schedule.
    """

    distribution_steps: int = 10
    attack_start_round: int = 10
    policy_train_end_round: int = 30
    inversion_lr: float = 0.05
    inversion_steps: int = 50
    reconstruction_batch_size: int = 8
    tv_weight: float = 0.02
    denoiser_noise_std: float = 0.3
    denoiser_epochs: int = 2
    seed_samples: int = 128
    proxy_buffer_limit: int = 2048
    state_tail_layers: int = 2
    state_include_num_attacker: bool = True
    simulator_horizon: int = 8
    episodes_per_observation: int = 1
    replay_capacity: int = 50_000
    td3_batch_size: int = 128
    policy_lr: float = 1e-7
    td3_gamma: float = 1.0
    td3_exploration_noise: float = 0.1
    td3_policy_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_tau: float = 0.005
    td3_policy_delay: int = 2
    td3_train_freq_steps: int = 5
    local_search_batch_size: int = 200
    gamma_max: float = 10.0
    krum_max_updates: int = 20
    robust_max_updates: int = 50
    attacker_local_lr: float = 0.05
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

    def action_dim(self, defense_type: str) -> int:
        return 3 if defense_type.lower() == "fltrust" else 2

    def action_bounds(self, defense_type: str) -> tuple[np.ndarray, np.ndarray]:
        dim = self.action_dim(defense_type)
        low = -np.ones(dim, dtype=np.float32)
        high = np.ones(dim, dtype=np.float32)
        return low, high

    def decode_action(self, action: np.ndarray, defense_type: str) -> DecodedAction:
        action_arr = np.asarray(action, dtype=np.float32)
        dim = self.action_dim(defense_type)
        if action_arr.shape[0] < dim:
            padded = np.zeros(dim, dtype=np.float32)
            padded[: action_arr.shape[0]] = action_arr
            action_arr = padded
        action_arr = np.clip(action_arr[:dim], -1.0, 1.0)
        defense = defense_type.lower()
        if defense == "fltrust":
            local_search_lr = float(action_arr[0]) * self.fltrust_lr_scale + self.fltrust_lr_center
            local_steps = int(round(float(action_arr[1]) * self.robust_steps_scale + self.robust_steps_center))
            lambda_stealth = float(action_arr[2]) * self.fltrust_alpha_scale + self.fltrust_alpha_center
            gamma_scale = 1.0
        elif defense in {"clipped_median", "median", "trimmed_mean", "geometric_median"}:
            gamma_scale = float(action_arr[0]) * self.clipped_gamma_scale + self.clipped_gamma_center
            local_steps = int(round(float(action_arr[1]) * self.clipped_steps_scale + self.clipped_steps_center))
            lambda_stealth = 0.0
            local_search_lr = self.attacker_local_lr
        else:
            gamma_scale = float(action_arr[0]) * self.robust_gamma_scale + self.robust_gamma_center
            local_steps = int(round(float(action_arr[1]) * self.robust_steps_scale + self.robust_steps_center))
            lambda_stealth = 0.0
            local_search_lr = self.attacker_local_lr
        return DecodedAction(
            gamma_scale=gamma_scale,
            local_steps=max(1, local_steps),
            lambda_stealth=min(1.0, max(0.0, lambda_stealth)),
            local_search_lr=max(1e-4, float(local_search_lr)),
        )

    def format_state(self, model: nn.Module, num_attackers: int, max_attackers: int) -> dict[str, Any]:
        compressed, _ = get_compressed_state(model, num_tail_layers=self.state_tail_layers)
        return {
            "pram": compressed.astype(np.float32),
            "num_attacker": int(max(0, min(max_attackers, num_attackers))),
        }

    def flatten_state(self, state: dict[str, Any], max_attackers: int) -> np.ndarray:
        pram = np.asarray(state["pram"], dtype=np.float32).reshape(-1)
        if not self.state_include_num_attacker:
            return pram
        attacker_norm = 0.0
        if max_attackers > 0:
            attacker_norm = 2.0 * (int(state["num_attacker"]) / max_attackers) - 1.0
        return np.concatenate([pram, np.asarray([attacker_norm], dtype=np.float32)], axis=0)


def local_search_update(
    *,
    model_template: nn.Module,
    old_weights: Weights,
    proxy_buffer: ProxyDatasetBuffer,
    device: torch.device,
    fl_lr: float,
    steps: int,
    gamma_scale: float,
    lambda_stealth: float,
    search_batch_size: int,
    state_tail_layers: int,
) -> Weights:
    if steps <= 0 or gamma_scale <= 0:
        return [layer.copy() for layer in old_weights]
    model = _build_model_from_template(model_template, old_weights, device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=fl_lr)
    origin_tail = [param.detach().clone() for _, param in _tail_named_parameters(model, state_tail_layers)]
    benign_x, benign_y = proxy_buffer.sample(search_batch_size, device)
    benign_loss = F.cross_entropy(model(benign_x), benign_y)
    benign_tail_grads = torch.autograd.grad(benign_loss, [param for _, param in _tail_named_parameters(model, state_tail_layers)], retain_graph=False)
    benign_grad_vec = _vectorize_tensors([grad.detach() for grad in benign_tail_grads])
    for _ in range(steps):
        images, labels = proxy_buffer.sample(search_batch_size, device)
        logits = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(ce_loss):
            break
        tail_params = [param for _, param in _tail_named_parameters(model, state_tail_layers)]
        if lambda_stealth > 0:
            update_vec = _vectorize_tensors([origin - current for origin, current in zip(origin_tail, tail_params)])
            cosine = F.cosine_similarity(update_vec.unsqueeze(0), benign_grad_vec.unsqueeze(0)).mean()
        else:
            cosine = torch.zeros((), device=device, dtype=ce_loss.dtype)
        objective = (1.0 - lambda_stealth) * ce_loss + lambda_stealth * cosine
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.data.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
    adv_weights = _capture_weights(model)
    crafted = []
    for old_layer, adv_layer in zip(old_weights, adv_weights):
        delta = np.nan_to_num(adv_layer - old_layer, nan=0.0, posinf=0.0, neginf=0.0)
        crafted.append(old_layer + gamma_scale * delta)
    return crafted


def _train_proxy_model(
    *,
    model_template: nn.Module,
    old_weights: Weights,
    proxy_buffer: ProxyDatasetBuffer,
    device: torch.device,
    lr: float,
    steps: int,
    batch_size: int,
) -> Weights:
    model = _build_model_from_template(model_template, old_weights, device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(max(1, steps)):
        images, labels = proxy_buffer.sample(batch_size, device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.data.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
    return _capture_weights(model)


class PaperRLAttacker:
    """Stateful RL attacker that plugs into the sandbox attack interface."""

    def __init__(self, config: Optional[RLAttackerConfig] = None) -> None:
        self.config = config or RLAttackerConfig()
        self.distribution_learner = GradientDistributionLearner(self.config)
        self.policy: Optional[TD3Agent] = None
        self.replay = ReplayBuffer(capacity=self.config.replay_capacity)
        self.model_template: Optional[nn.Module] = None
        self.device = torch.device("cpu")
        self.fl_config = None
        self.defender: Optional[AggregationDefender] = None
        self.prev_weights: Optional[Weights] = None
        self.prev_round_idx: Optional[int] = None
        self.latest_policy_weights: Optional[Weights] = None
        self.ready = False

    def observe_round(self, ctx: RoundContext) -> None:
        if ctx.model is None or ctx.device is None:
            return
        if self.model_template is None:
            self.model_template = copy.deepcopy(ctx.model).cpu()
        self.device = ctx.device
        self.fl_config = ctx.fl_config
        if self.defender is None:
            self.defender = AggregationDefender(
                defense_type=ctx.defense_type,
                krum_attackers=getattr(ctx.fl_config, "krum_attackers", 1),
                multi_krum_selected=getattr(ctx.fl_config, "multi_krum_selected", None),
                clipped_median_norm=getattr(ctx.fl_config, "clipped_median_norm", 2.0),
                trimmed_mean_ratio=getattr(ctx.fl_config, "trimmed_mean_ratio", 0.2),
                geometric_median_iters=getattr(ctx.fl_config, "geometric_median_iters", 10),
            )
        self.distribution_learner.initialize_from_loader(ctx.all_attacker_train_iter or ctx.attacker_train_iter, self.device)
        self.latest_policy_weights = [layer.copy() for layer in ctx.old_weights]
        if (
            self.prev_weights is not None
            and ctx.round_idx <= self.config.distribution_steps
            and self.model_template is not None
        ):
            round_gap = ctx.round_idx - (self.prev_round_idx or ctx.round_idx - 1)
            self.distribution_learner.observe_server_update(
                previous_weights=self.prev_weights,
                current_weights=ctx.old_weights,
                model_template=self.model_template,
                server_lr=max(1e-8, ctx.server_lr),
                round_gap=round_gap,
            )
        self.prev_weights = [layer.copy() for layer in ctx.old_weights]
        self.prev_round_idx = ctx.round_idx
        self.ready = len(self.distribution_learner.buffer) > 0 and self.model_template is not None and self.defender is not None
        if self.ready and ctx.round_idx <= self.config.policy_train_end_round:
            self._train_policy()

    def execute(self, ctx: RoundContext, attacker_action: Optional[np.ndarray] = None) -> list[Weights]:
        num_attackers = len(ctx.selected_attacker_ids)
        if num_attackers == 0:
            return []
        if not self.ready or ctx.round_idx < self.config.attack_start_round or self.model_template is None:
            return [[layer.copy() for layer in ctx.old_weights] for _ in range(num_attackers)]
        state = self._current_state(ctx)
        if attacker_action is not None:
            action = np.asarray(attacker_action, dtype=np.float32)
        elif self.policy is not None:
            action = self.policy.act(state, noise_scale=0.0)
        else:
            low, high = self.config.action_bounds(ctx.defense_type)
            action = 0.5 * (low + high)
        decoded = self.config.decode_action(action, ctx.defense_type)
        crafted = self._craft_malicious_weights(ctx.old_weights, decoded, ctx.defense_type)
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]

    def _craft_malicious_weights(
        self,
        old_weights: Weights,
        decoded: DecodedAction,
        defense_type: str,
    ) -> Weights:
        if defense_type.lower() == "fltrust":
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

    def _current_state(self, ctx: RoundContext) -> np.ndarray:
        if self.model_template is None:
            raise RuntimeError("RL attacker state requested before initialization")
        model = _build_model_from_template(self.model_template, ctx.old_weights, self.device)
        num_attackers = max(0, getattr(ctx.fl_config, "num_attackers", 1))
        num_clients = max(1, getattr(ctx.fl_config, "num_clients", 1))
        subsample_rate = float(getattr(ctx.fl_config, "subsample_rate", 1.0))
        # Match the ceiling used by SimulatedFLEnv so the num_attacker scalar is
        # normalized identically during policy training and live execution.
        max_attackers = max(1, min(num_attackers, max(1, int(num_clients * subsample_rate))))
        state_dict = self.config.format_state(model, len(ctx.selected_attacker_ids), max_attackers)
        return self.config.flatten_state(state_dict, max_attackers)

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
        sim_env = SimulatedFLEnv(
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

    def _infer_state_dim(self) -> int:
        if self.model_template is None or self.latest_policy_weights is None:
            raise RuntimeError("Cannot infer RL attacker state dim before initialization")
        model = _build_model_from_template(self.model_template, self.latest_policy_weights, self.device)
        compressed, _ = get_compressed_state(model, num_tail_layers=self.config.state_tail_layers)
        return int(compressed.shape[0] + (1 if self.config.state_include_num_attacker else 0))
