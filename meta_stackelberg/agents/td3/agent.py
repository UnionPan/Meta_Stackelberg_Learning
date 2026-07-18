"""Deterministic CPU-compatible TD3 with exact role freeze snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
import torch

from meta_stackelberg.agents.td3.networks import TD3Actor, TD3Critic
from meta_stackelberg.agents.td3.replay import TD3Batch


@dataclass(frozen=True)
class TD3UpdateStats:
    critic_loss: float
    actor_loss: float | None
    actor_updated: bool
    q_mean: float


@dataclass(frozen=True)
class TD3Snapshot:
    schema_version: int
    role: str
    actor: dict
    actor_target: dict
    critic1: dict
    critic2: dict
    critic1_target: dict
    critic2_target: dict
    actor_optimizer: dict
    critic_optimizer: dict
    update_count: int
    torch_rng_state: torch.Tensor
    numpy_rng_state: dict


class TD3FreezeGuard:
    def __init__(self, agent: TD3Agent, replays=()) -> None:
        self.agent = agent
        self.role = agent.role
        self.fingerprint = agent.fingerprint()
        self.replays = tuple(replays)
        if any(replay.role != agent.role for replay in self.replays):
            raise ValueError('freeze replay role must match policy role')
        self.replay_fingerprints = tuple(
            replay.fingerprint() for replay in self.replays
        )

    def verify(self) -> None:
        if self.agent.fingerprint() != self.fingerprint:
            raise RuntimeError(f'{self.role} freeze fingerprint changed')
        if tuple(replay.fingerprint() for replay in self.replays) != self.replay_fingerprints:
            raise RuntimeError(f'{self.role} replay freeze fingerprint changed')


class TD3Agent:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        role: str,
        seed: int,
        hidden_sizes: tuple[int, ...],
        learning_rate: float,
        gamma: float,
        tau: float,
        policy_delay: int,
        target_policy_noise: float,
        noise_clip: float,
        device: str | torch.device = 'cpu',
    ) -> None:
        if role not in {'defender', 'attacker'}:
            raise ValueError('role must be defender or attacker')
        self.role = role
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.noise_clip = float(noise_clip)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {self.device} is not available')
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.actor = TD3Actor(obs_dim, action_dim, hidden_sizes).to(self.device)
            self.critic1 = TD3Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
            self.critic2 = TD3Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            tuple(self.critic1.parameters()) + tuple(self.critic2.parameters()),
            lr=learning_rate,
        )
        self.update_count = 0
        self._torch_rng = torch.Generator(device=self.device).manual_seed(seed + 1)
        self._numpy_rng = np.random.default_rng(seed + 2)

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        exploration_noise: float = 0.1,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.obs_dim,) or not np.all(np.isfinite(obs)):
            raise ValueError('observation has invalid shape or values')
        with torch.no_grad():
            action = self.actor(
                torch.from_numpy(obs).to(self.device).unsqueeze(0),
            )[0].detach().cpu().numpy()
        if not deterministic:
            action = action + self._numpy_rng.normal(
                0.0, exploration_noise, size=self.action_dim,
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def sample_uniform_action(self) -> np.ndarray:
        """Sample the SB3-style pre-learning action over the full action box."""
        return self._numpy_rng.uniform(
            -1.0, 1.0, size=self.action_dim,
        ).astype(np.float32)

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        return rewards + self.gamma * (1.0 - dones) * target_q

    def update(
        self,
        batch: TD3Batch,
        *,
        actor_logit_l2: float = 0.0,
        actor_logit_l2_mask: tuple[float, ...] | None = None,
    ) -> TD3UpdateStats:
        if not math.isfinite(actor_logit_l2) or actor_logit_l2 < 0:
            raise ValueError('actor_logit_l2 must be non-negative and finite')
        if actor_logit_l2_mask is not None and (
            len(actor_logit_l2_mask) != self.action_dim
            or any(
                not math.isfinite(value) or value < 0
                for value in actor_logit_l2_mask
            )
        ):
            raise ValueError(
                'actor_logit_l2_mask must contain one non-negative finite '
                'weight per action dimension'
            )
        observations = torch.from_numpy(batch.observations).to(self.device).float()
        actions = torch.from_numpy(batch.actions).to(self.device).float()
        rewards = torch.from_numpy(batch.rewards).to(self.device).float()
        next_observations = torch.from_numpy(
            batch.next_observations,
        ).to(self.device).float()
        dones = torch.from_numpy(batch.dones).to(self.device).float()
        with torch.no_grad():
            noise = torch.randn(
                (len(observations), self.action_dim), generator=self._torch_rng,
                device=self.device,
            ) * self.target_policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1.0, 1.0)
            target_q = torch.minimum(
                self.critic1_target(next_observations, next_actions),
                self.critic2_target(next_observations, next_actions),
            )
            td_target = self.compute_td_target(rewards, dones, target_q)
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        critic_loss = torch.nn.functional.mse_loss(q1, td_target) + torch.nn.functional.mse_loss(q2, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.update_count += 1
        actor_loss_value = None
        actor_updated = self.update_count % self.policy_delay == 0
        if actor_updated:
            actor_loss = -self.critic1(observations, self.actor(observations)).mean()
            if actor_logit_l2:
                # A tanh actor initialized at |logit| >> 1 receives virtually
                # no policy gradient.  This optional online-only penalty moves
                # it back into a trainable region without changing the default
                # TD3 objective or any existing checkpoint.
                squared_logits = self.actor.logits(observations).square()
                if actor_logit_l2_mask is not None:
                    mask = torch.tensor(
                        actor_logit_l2_mask,
                        dtype=squared_logits.dtype,
                        device=self.device,
                    )
                    squared_logits = squared_logits * mask
                actor_loss = actor_loss + float(actor_logit_l2) * squared_logits.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
            actor_loss_value = float(actor_loss.detach())
        return TD3UpdateStats(
            critic_loss=float(critic_loss.detach()),
            actor_loss=actor_loss_value,
            actor_updated=actor_updated,
            q_mean=float(torch.minimum(q1, q2).detach().mean()),
        )

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def snapshot(self) -> TD3Snapshot:
        return TD3Snapshot(
            1,
            self.role,
            _state(self.actor),
            _state(self.actor_target),
            _state(self.critic1),
            _state(self.critic2),
            _state(self.critic1_target),
            _state(self.critic2_target),
            copy.deepcopy(self.actor_optimizer.state_dict()),
            copy.deepcopy(self.critic_optimizer.state_dict()),
            self.update_count,
            self._torch_rng.get_state().clone(),
            copy.deepcopy(self._numpy_rng.bit_generator.state),
        )

    def restore(self, snapshot: TD3Snapshot) -> None:
        if not isinstance(snapshot, TD3Snapshot) or snapshot.schema_version != 1:
            raise ValueError('unknown TD3 snapshot schema')
        if snapshot.role != self.role:
            raise ValueError('TD3 snapshot role mismatch')
        for module, state in (
            (self.actor, snapshot.actor),
            (self.actor_target, snapshot.actor_target),
            (self.critic1, snapshot.critic1),
            (self.critic2, snapshot.critic2),
            (self.critic1_target, snapshot.critic1_target),
            (self.critic2_target, snapshot.critic2_target),
        ):
            module.load_state_dict(copy.deepcopy(state))
        self.actor_optimizer.load_state_dict(copy.deepcopy(snapshot.actor_optimizer))
        self.critic_optimizer.load_state_dict(copy.deepcopy(snapshot.critic_optimizer))
        self.update_count = snapshot.update_count
        self._torch_rng.set_state(snapshot.torch_rng_state.detach().cpu().clone())
        self._numpy_rng.bit_generator.state = copy.deepcopy(snapshot.numpy_rng_state)

    def fingerprint(self) -> str:
        digest = sha256()
        _hash_value(digest, self.snapshot())
        return digest.hexdigest()

    def freeze_guard(self, *replays) -> TD3FreezeGuard:
        return TD3FreezeGuard(self, replays)

    def clone(self) -> TD3Agent:
        """Return an isolated task-policy copy including optimizer and RNG state."""
        return copy.deepcopy(self)

    def set_learning_rate(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = float(learning_rate)


def _state(module: torch.nn.Module) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _hash_value(digest, value) -> None:
    if isinstance(value, TD3Snapshot):
        for field in value.__dataclass_fields__:
            _hash_value(digest, getattr(value, field))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(repr(value).encode())
