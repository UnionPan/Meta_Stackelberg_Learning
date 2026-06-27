"""Tianshou-stable SAC adapter used for few-shot Meta-SG evaluation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from gymnasium import spaces

from tianshou.algorithm import SAC
from tianshou.algorithm.modelfree.sac import SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.torch_utils import policy_within_training_step

from .config import TD3Config
from .replay_buffer import ReplayBuffer
from .td3 import TD3Agent, _set_optimizer_lr


class SACAgent:
    """Project-local continuous SAC adapter with the TD3Agent surface needed for evaluation."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: TD3Config,
        device: Optional[torch.device] = None,
        *,
        alpha: float = 0.2,
        conditioned_sigma: bool = False,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.alpha = float(alpha)
        self.conditioned_sigma = bool(conditioned_sigma)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )
        self.actor = ContinuousActorProbabilistic(
            preprocess_net=Net(
                state_shape=obs_dim,
                hidden_sizes=(config.hidden_dim, config.hidden_dim),
            ),
            action_shape=(act_dim,),
            hidden_sizes=(),
            max_action=1.0,
            unbounded=True,
            conditioned_sigma=self.conditioned_sigma,
        ).to(self.device)
        self.critic1 = ContinuousCritic(
            preprocess_net=Net(
                state_shape=obs_dim,
                action_shape=act_dim,
                hidden_sizes=(config.hidden_dim, config.hidden_dim),
                concat=True,
            ),
            hidden_sizes=(),
        ).to(self.device)
        self.critic2 = ContinuousCritic(
            preprocess_net=Net(
                state_shape=obs_dim,
                action_shape=act_dim,
                hidden_sizes=(config.hidden_dim, config.hidden_dim),
                concat=True,
            ),
            hidden_sizes=(),
        ).to(self.device)

        self.policy = SACPolicy(
            actor=self.actor,
            exploration_noise=None,
            deterministic_eval=True,
            action_scaling=True,
            action_space=self.action_space,
            observation_space=self.observation_space,
        )
        self.algorithm = SAC(
            policy=self.policy,
            policy_optim=AdamOptimizerFactory(lr=config.policy_lr),
            critic=self.critic1,
            critic_optim=AdamOptimizerFactory(lr=config.critic_lr),
            critic2=self.critic2,
            critic2_optim=AdamOptimizerFactory(lr=config.critic_lr),
            tau=config.tau,
            gamma=config.gamma,
            alpha=float(alpha),
            deterministic_eval=True,
        ).to(self.device)
        self._step = 0

    @classmethod
    def from_td3(
        cls,
        td3_agent: TD3Agent,
        *,
        alpha: float = 0.2,
        conditioned_sigma: bool = False,
    ) -> "SACAgent":
        agent = cls(
            td3_agent.obs_dim,
            td3_agent.act_dim,
            td3_agent.cfg,
            td3_agent.device,
            alpha=alpha,
            conditioned_sigma=conditioned_sigma,
        )
        agent._load_td3_networks(td3_agent)
        return agent

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        action = self.policy.compute_action(np.asarray(obs, dtype=np.float32))
        action = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        if noise > 0.0:
            action += np.random.normal(0, noise, size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {}

        batch_size = min(self.cfg.batch_size, len(buffer))
        with policy_within_training_step(self.policy):
            stats = self.algorithm.update(buffer.tianshou_buffer, sample_size=batch_size)
        self._step += 1
        critic1_loss = float(stats.critic1_loss)
        critic2_loss = float(stats.critic2_loss)
        result = {
            "critic_loss": critic1_loss + critic2_loss,
            "actor_loss": float(stats.actor_loss),
            "critic1_loss": critic1_loss,
            "critic2_loss": critic2_loss,
            "alpha": float(stats.alpha) if stats.alpha is not None else float("nan"),
        }
        if stats.alpha_loss is not None:
            result["alpha_loss"] = float(stats.alpha_loss)
        return result

    def set_learning_rates(
        self,
        *,
        policy_lr: float | None = None,
        critic_lr: float | None = None,
    ) -> None:
        if policy_lr is not None:
            _set_optimizer_lr(self.algorithm.policy_optim, policy_lr)
        if critic_lr is not None:
            _set_optimizer_lr(self.algorithm.critic_optim, critic_lr)
            _set_optimizer_lr(self.algorithm.critic2_optim, critic_lr)

    def clone(self) -> "SACAgent":
        new = SACAgent(
            self.obs_dim,
            self.act_dim,
            self.cfg,
            self.device,
            alpha=self.alpha,
            conditioned_sigma=self.conditioned_sigma,
        )
        new.actor.load_state_dict(self.actor.state_dict())
        new.critic1.load_state_dict(self.critic1.state_dict())
        new.critic2.load_state_dict(self.critic2.state_dict())
        new.algorithm._lagged_networks.full_parameter_update()
        new._step = self._step
        return new

    def _load_td3_networks(self, td3_agent: TD3Agent) -> None:
        td3_actor_sd = td3_agent.actor.state_dict()
        sac_actor_sd = self.actor.state_dict()
        for key, value in td3_actor_sd.items():
            if key.startswith("preprocess."):
                sac_actor_sd[key] = value.clone()
            elif key.startswith("last."):
                sac_actor_sd[f"mu.{key[len('last.'):]}"] = value.clone()
        self.actor.load_state_dict(sac_actor_sd)
        self.critic1.load_state_dict(td3_agent.critic1.state_dict())
        self.critic2.load_state_dict(td3_agent.critic2.state_dict())
        self.algorithm._lagged_networks.full_parameter_update()
