"""
Twin Delayed DDPG (TD3) agent.
Used for both the defender policy π_D(s; θ) and attacker policy π_ξ(s; φ).
Paper §Appendix C-A: TD3 with policy_lr=0.001, batch_size=256, γ=0.99.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TD3Config
from .policies import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """
    TD3 agent with:
      - Delayed policy updates (every policy_delay critic steps)
      - Target policy smoothing
      - Polyak target network updates
      - clone() / get_params() / set_params() for Reptile meta-updates
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TD3Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config
        self.device = device or torch.device("cpu")

        self.actor = Actor(obs_dim, act_dim, config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(obs_dim, act_dim, config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self._step = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """Select action with optional exploration noise."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise > 0.0:
            action += np.random.normal(0, noise, size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """One TD3 gradient update step. Returns loss dict."""
        if len(buffer) == 0:
            return {}

        batch_size = min(self.cfg.batch_size, len(buffer))
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        nobs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # --- Critic update ---
        with torch.no_grad():
            noise = (
                torch.randn_like(act_t) * self.cfg.target_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(nobs_t) + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(nobs_t, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = rew_t + (1.0 - done_t) * self.cfg.gamma * q_next

        q_target_mean = float(target_q.mean().item())

        q1, q2 = self.critic(obs_t, act_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self._step += 1
        actor_loss_val = float("nan")

        # --- Delayed actor update ---
        if self._step % self.cfg.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs_t, self.actor(obs_t)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            # Polyak target updates
            _polyak_update(self.actor, self.actor_target, self.cfg.tau)
            _polyak_update(self.critic, self.critic_target, self.cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
            "q_mean": q_target_mean,
        }

    # ------------------------------------------------------------------
    # Reptile meta-learning helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return copies of actor + critic parameters for Reptile."""
        return {
            **{f"actor.{k}": v.clone() for k, v in self.actor.state_dict().items()},
            **{f"critic.{k}": v.clone() for k, v in self.critic.state_dict().items()},
        }

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load actor + critic parameters (e.g., after Reptile meta-update)."""
        actor_sd = {k[len("actor."):]: v for k, v in params.items() if k.startswith("actor.")}
        critic_sd = {k[len("critic."):]: v for k, v in params.items() if k.startswith("critic.")}
        if actor_sd:
            self.actor.load_state_dict(actor_sd)
            self.actor_target = copy.deepcopy(self.actor)
        if critic_sd:
            self.critic.load_state_dict(critic_sd)
            self.critic_target = copy.deepcopy(self.critic)

    def clone(self) -> "TD3Agent":
        """Return a deep copy — used for per-attack-type Reptile adaptation."""
        new = TD3Agent(self.obs_dim, self.act_dim, self.cfg, self.device)
        new.set_params(self.get_params())
        new._step = self._step
        return new

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "step": self._step,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self._step = ckpt.get("step", 0)


def _polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for s_p, t_p in zip(source.parameters(), target.parameters()):
            t_p.data.mul_(1.0 - tau).add_(tau * s_p.data)
