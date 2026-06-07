"""Multi-agent TD3 for attacker/defender co-learning."""
from __future__ import annotations

import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_sg.learning.config import TD3Config


JointBatch = namedtuple(
    "JointBatch",
    [
        "obs",
        "defender_action",
        "attacker_action",
        "defender_reward",
        "attacker_reward",
        "next_obs",
        "done",
    ],
)

_JointTransition = namedtuple(
    "_JointTransition",
    [
        "obs",
        "defender_action",
        "attacker_action",
        "defender_reward",
        "attacker_reward",
        "next_obs",
        "done",
    ],
)


class JointReplayBuffer:
    """Replay buffer storing one shared-state transition with both actions/rewards."""

    def __init__(self, capacity: int, obs_dim: int, defender_act_dim: int, attacker_act_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.defender_act_dim = int(defender_act_dim)
        self.attacker_act_dim = int(attacker_act_dim)
        self._storage = deque(maxlen=self.capacity)
        self._rng = random.Random(0)

    def add(
        self,
        obs: np.ndarray,
        defender_action: np.ndarray,
        attacker_action: np.ndarray,
        defender_reward: float,
        attacker_reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._storage.append(
            _JointTransition(
                obs=np.asarray(obs, dtype=np.float32).reshape(self.obs_dim),
                defender_action=np.asarray(defender_action, dtype=np.float32).reshape(self.defender_act_dim),
                attacker_action=np.asarray(attacker_action, dtype=np.float32).reshape(self.attacker_act_dim),
                defender_reward=float(defender_reward),
                attacker_reward=float(attacker_reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> JointBatch:
        batch_size = min(int(batch_size), len(self._storage))
        if batch_size <= 0:
            raise ValueError("Cannot sample from an empty JointReplayBuffer")
        batch = self._rng.sample(list(self._storage), batch_size)
        return JointBatch(
            obs=_tensor([item.obs for item in batch], device),
            defender_action=_tensor([item.defender_action for item in batch], device),
            attacker_action=_tensor([item.attacker_action for item in batch], device),
            defender_reward=_tensor([[item.defender_reward] for item in batch], device),
            attacker_reward=_tensor([[item.attacker_reward] for item in batch], device),
            next_obs=_tensor([item.next_obs for item in batch], device),
            done=_tensor([[float(item.done)] for item in batch], device),
        )

    def __len__(self) -> int:
        return len(self._storage)


class _Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _TwinCritic(nn.Module):
    def __init__(self, joint_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = self._network(joint_dim, hidden_dim)
        self.q2 = self._network(joint_dim, hidden_dim)

    def forward(self, joint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(joint), self.q2(joint)

    def q1_value(self, joint: torch.Tensor) -> torch.Tensor:
        return self.q1(joint)

    @staticmethod
    def _network(joint_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


@dataclass
class _AgentNetworks:
    actor: _Actor
    actor_target: _Actor
    critic: _TwinCritic
    critic_target: _TwinCritic
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer


class MATD3AgentPair:
    """Two-agent MATD3: decentralized actors, centralized twin critics."""

    def __init__(
        self,
        obs_dim: int,
        defender_act_dim: int,
        attacker_act_dim: int,
        config: TD3Config,
        device: torch.device | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.defender_act_dim = int(defender_act_dim)
        self.attacker_act_dim = int(attacker_act_dim)
        self.config = config
        self.device = device or torch.device("cpu")
        joint_dim = self.obs_dim + self.defender_act_dim + self.attacker_act_dim
        self.defender = self._build_agent(self.defender_act_dim, joint_dim)
        self.attacker = self._build_agent(self.attacker_act_dim, joint_dim)
        self._hard_update_all()
        self._update_step = 0

    def get_actions(self, obs: np.ndarray, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, self.obs_dim)
        with torch.no_grad():
            defender_action = self.defender.actor(obs_t).cpu().numpy()[0]
            attacker_action = self.attacker.actor(obs_t).cpu().numpy()[0]
        if noise > 0.0:
            defender_action += np.random.normal(0.0, noise, size=defender_action.shape)
            attacker_action += np.random.normal(0.0, noise, size=attacker_action.shape)
        return (
            np.clip(defender_action, -1.0, 1.0).astype(np.float32),
            np.clip(attacker_action, -1.0, 1.0).astype(np.float32),
        )

    def update(
        self,
        buffer: JointReplayBuffer,
        gradient_steps: int = 1,
        *,
        update_defender: bool = True,
        update_attacker: bool = True,
        fixed_defender_action: np.ndarray | None = None,
    ) -> Dict[str, float]:
        if len(buffer) <= 0:
            return {}
        stats: Dict[str, float] = {}
        for _ in range(max(1, int(gradient_steps))):
            batch = buffer.sample(self.config.batch_size, self.device)
            stats.update(
                self._update_once(
                    batch,
                    update_defender=update_defender,
                    update_attacker=update_attacker,
                    fixed_defender_action=fixed_defender_action,
                )
            )
        return stats

    def _update_once(
        self,
        batch: JointBatch,
        *,
        update_defender: bool = True,
        update_attacker: bool = True,
        fixed_defender_action: np.ndarray | None = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            if fixed_defender_action is None:
                next_defender_action = self.defender.actor_target(batch.next_obs)
                next_defender_action = _smooth_target_action(
                    next_defender_action,
                    self.config.target_noise,
                    self.config.noise_clip,
                )
            else:
                fixed_defender = torch.as_tensor(
                    np.asarray(fixed_defender_action, dtype=np.float32),
                    device=self.device,
                ).reshape(1, self.defender_act_dim)
                next_defender_action = fixed_defender.expand(batch.next_obs.shape[0], -1)
            if update_attacker:
                next_attacker_action = self.attacker.actor_target(batch.next_obs)
                next_attacker_action = _smooth_target_action(
                    next_attacker_action,
                    self.config.target_noise,
                    self.config.noise_clip,
                )
            else:
                next_attacker_action = batch.attacker_action
            next_joint = self._joint(batch.next_obs, next_defender_action, next_attacker_action)

        stats = {}
        if update_defender:
            stats.update(
                self._update_critic(
                    name="defender",
                    agent=self.defender,
                    joint=self._joint(batch.obs, batch.defender_action, batch.attacker_action),
                    next_joint=next_joint,
                    reward=batch.defender_reward,
                    done=batch.done,
                )
            )
        if update_attacker:
            stats.update(
                self._update_critic(
                    name="attacker",
                    agent=self.attacker,
                    joint=self._joint(batch.obs, batch.defender_action, batch.attacker_action),
                    next_joint=next_joint,
                    reward=batch.attacker_reward,
                    done=batch.done,
                )
            )

        if self._update_step % max(1, int(self.config.policy_delay)) == 0:
            if update_defender:
                defender_policy_action = self.defender.actor(batch.obs)
                stats["defender_actor_loss"] = self._update_actor(
                    self.defender,
                    self._joint(batch.obs, defender_policy_action, batch.attacker_action.detach()),
                )
                _soft_update(self.defender.actor_target, self.defender.actor, self.config.tau)
                _soft_update(self.defender.critic_target, self.defender.critic, self.config.tau)
            if update_attacker:
                attacker_policy_action = self.attacker.actor(batch.obs)
                defender_context = (
                    torch.as_tensor(
                        np.asarray(fixed_defender_action, dtype=np.float32),
                        device=self.device,
                    )
                    .reshape(1, self.defender_act_dim)
                    .expand(batch.obs.shape[0], -1)
                    if fixed_defender_action is not None
                    else batch.defender_action.detach()
                )
                stats["attacker_actor_loss"] = self._update_actor(
                    self.attacker,
                    self._joint(batch.obs, defender_context, attacker_policy_action),
                )
                _soft_update(self.attacker.actor_target, self.attacker.actor, self.config.tau)
                _soft_update(self.attacker.critic_target, self.attacker.critic, self.config.tau)

        self._update_step += 1
        return stats

    def _update_critic(
        self,
        *,
        name: str,
        agent: _AgentNetworks,
        joint: torch.Tensor,
        next_joint: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            target_q1, target_q2 = agent.critic_target(next_joint)
            target_q = torch.min(target_q1, target_q2)
            y = reward + self.config.gamma * (1.0 - done) * target_q
        q1, q2 = agent.critic(joint)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        agent.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        agent.critic_opt.step()
        return {f"{name}_critic_loss": float(critic_loss.detach().cpu().item())}

    def _update_actor(self, agent: _AgentNetworks, joint: torch.Tensor) -> float:
        actor_loss = -agent.critic.q1_value(joint).mean()
        agent.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        agent.actor_opt.step()
        return float(actor_loss.detach().cpu().item())

    def _build_agent(self, act_dim: int, joint_dim: int) -> _AgentNetworks:
        actor = _Actor(self.obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        actor_target = _Actor(self.obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        critic = _TwinCritic(joint_dim, self.config.hidden_dim).to(self.device)
        critic_target = _TwinCritic(joint_dim, self.config.hidden_dim).to(self.device)
        return _AgentNetworks(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            actor_opt=torch.optim.Adam(actor.parameters(), lr=self.config.policy_lr),
            critic_opt=torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr),
        )

    def _joint(
        self,
        obs: torch.Tensor,
        defender_action: torch.Tensor,
        attacker_action: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([obs, defender_action, attacker_action], dim=-1)

    def _hard_update_all(self) -> None:
        for agent in (self.defender, self.attacker):
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())

    def _soft_update_all(self) -> None:
        for agent in (self.defender, self.attacker):
            _soft_update(agent.actor_target, agent.actor, self.config.tau)
            _soft_update(agent.critic_target, agent.critic, self.config.tau)


def _tensor(values, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def _smooth_target_action(action: torch.Tensor, target_noise: float, noise_clip: float) -> torch.Tensor:
    if target_noise <= 0.0:
        return torch.clamp(action, -1.0, 1.0)
    noise = torch.randn_like(action) * float(target_noise)
    noise = torch.clamp(noise, -float(noise_clip), float(noise_clip))
    return torch.clamp(action + noise, -1.0, 1.0)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau)
            target_param.add_(tau * source_param)
