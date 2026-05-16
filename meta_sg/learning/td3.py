"""
Tianshou-stable TD3 agent adapter.

Tianshou v2 separates policies from learning algorithms.  This wrapper keeps
the project-local Meta-SG interface stable while delegating actor/critic
networks, target networks, optimizers, and TD3 updates to Tianshou.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from gymnasium import spaces

try:
    import tianshou
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError(
        "TD3Agent targets Tianshou stable v2.x. Install it in a Python >=3.11 "
        "environment with: pip install 'tianshou==2.0.1'"
    ) from exc

if int(tianshou.__version__.split(".", 1)[0]) < 2:
    raise ImportError(
        f"TD3Agent targets Tianshou stable v2.x, but found tianshou=={tianshou.__version__}. "
        "Tianshou v2 requires Python >=3.11; recreate the project environment with "
        "Python 3.11+ and install 'tianshou==2.0.1'."
    )

from tianshou.algorithm import TD3
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic
from tianshou.utils.torch_utils import policy_within_training_step

from .config import TD3Config
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Project-local adapter around Tianshou stable TD3.

    The Meta-SG trainer still expects get_action/update/clone/get_params/
    set_params/save/load, so those methods remain here while the learning
    implementation lives in Tianshou.
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

        self.actor = ContinuousActorDeterministic(
            preprocess_net=Net(
                state_shape=obs_dim,
                hidden_sizes=(config.hidden_dim, config.hidden_dim),
            ),
            action_shape=(act_dim,),
            hidden_sizes=(),
            max_action=1.0,
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

        self.policy = ContinuousDeterministicPolicy(
            actor=self.actor,
            exploration_noise=GaussianNoise(sigma=config.exploration_noise),
            action_space=self.action_space,
            observation_space=self.observation_space,
            action_scaling=True,
            action_bound_method="clip",
        )
        self.algorithm = TD3(
            policy=self.policy,
            policy_optim=AdamOptimizerFactory(lr=config.policy_lr),
            critic=self.critic1,
            critic_optim=AdamOptimizerFactory(lr=config.critic_lr),
            critic2=self.critic2,
            critic2_optim=AdamOptimizerFactory(lr=config.critic_lr),
            tau=config.tau,
            gamma=config.gamma,
            policy_noise=config.target_noise,
            update_actor_freq=config.policy_delay,
            noise_clip=config.noise_clip,
        ).to(self.device)
        self._step = 0

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """Select an action with optional Gaussian exploration noise."""
        action = self.policy.compute_action(np.asarray(obs, dtype=np.float32))
        action = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        if noise > 0.0:
            action += np.random.normal(0, noise, size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Run one Tianshou stable TD3 update. Returns project-style losses."""
        if len(buffer) == 0:
            return {}

        batch_size = min(self.cfg.batch_size, len(buffer))
        prev_cnt = self.algorithm._cnt
        with policy_within_training_step(self.policy):
            stats = self.algorithm.update(buffer.tianshou_buffer, sample_size=batch_size)
        self._step += 1

        actor_updated = prev_cnt % self.algorithm.update_actor_freq == 0
        actor_loss = stats.actor_loss if actor_updated else float("nan")
        critic1_loss = float(stats.critic1_loss)
        critic2_loss = float(stats.critic2_loss)

        return {
            "critic_loss": critic1_loss + critic2_loss,
            "actor_loss": float(actor_loss),
            "critic1_loss": critic1_loss,
            "critic2_loss": critic2_loss,
        }

    def set_learning_rates(
        self,
        *,
        policy_lr: float | None = None,
        critic_lr: float | None = None,
    ) -> None:
        """Override optimizer learning rates, mainly for cautious online adaptation."""
        if policy_lr is not None:
            _set_optimizer_lr(self.algorithm.policy_optim, policy_lr)
        if critic_lr is not None:
            _set_optimizer_lr(self.algorithm.critic_optim, critic_lr)
            _set_optimizer_lr(self.algorithm.critic2_optim, critic_lr)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return copies of trainable actor + critic parameters for Reptile."""
        return {
            **{f"actor.{k}": v.clone() for k, v in self.actor.state_dict().items()},
            **{f"critic1.{k}": v.clone() for k, v in self.critic1.state_dict().items()},
            **{f"critic2.{k}": v.clone() for k, v in self.critic2.state_dict().items()},
        }

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load trainable actor + critic parameters and refresh target networks."""
        actor_sd = {k[len("actor."):]: v for k, v in params.items() if k.startswith("actor.")}
        critic1_sd = {k[len("critic1."):]: v for k, v in params.items() if k.startswith("critic1.")}
        critic2_sd = {k[len("critic2."):]: v for k, v in params.items() if k.startswith("critic2.")}

        if actor_sd:
            self.actor.load_state_dict(actor_sd)
        if critic1_sd:
            self.critic1.load_state_dict(critic1_sd)
        if critic2_sd:
            self.critic2.load_state_dict(critic2_sd)
        self.algorithm._lagged_networks.full_parameter_update()

    def clone(self) -> "TD3Agent":
        """Return a copy used for per-attack-type Reptile adaptation."""
        new = TD3Agent(self.obs_dim, self.act_dim, self.cfg, self.device)
        new.set_params(self.get_params())
        new._step = self._step
        new.algorithm._cnt = self.algorithm._cnt
        new.algorithm._last = self.algorithm._last
        return new

    def save(self, path: str) -> None:
        torch.save(
            {
                "algorithm": self.algorithm.state_dict(),
                "step": self._step,
                "algorithm_cnt": self.algorithm._cnt,
                "algorithm_last": self.algorithm._last,
                "tianshou_version": tianshou.__version__,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "algorithm" not in ckpt:
            raise ValueError(
                "This checkpoint was not created by the Tianshou stable TD3Agent. "
                "Retrain or convert the old handwritten-TD3 checkpoint first."
            )
        self.algorithm.load_state_dict(ckpt["algorithm"])
        self._step = ckpt.get("step", 0)
        self.algorithm._cnt = ckpt.get("algorithm_cnt", self.algorithm._cnt)
        self.algorithm._last = ckpt.get("algorithm_last", self.algorithm._last)


def _set_optimizer_lr(optimizer_wrapper, lr: float) -> None:
    for group in optimizer_wrapper._optim.param_groups:
        group["lr"] = lr
