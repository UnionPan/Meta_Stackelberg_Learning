"""Shared Tianshou trainer utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.trainer import CollectStats, UpdateStats


class BaseTianshouTrainer:
    algorithm_name = "base"

    def __init__(self, config: RLAttackerConfig) -> None:
        from tianshou.data import ReplayBuffer

        self.config = config
        self.replay = ReplayBuffer(size=config.replay_capacity)
        self.algorithm = None
        self.policy = None
        self.action_low: np.ndarray | None = None
        self.action_high: np.ndarray | None = None
        self.collect_steps = 0
        self.update_steps = 0
        self.last_loss = 0.0
        self.last_reward_mean = 0.0
        self.last_training_stats: dict[str, float] = {}

    def ensure_initialized(self, obs_space, action_space) -> None:
        self.action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if self.algorithm is None:
            self.algorithm = self._build_algorithm(obs_space, action_space)
            self.policy = self.algorithm.policy

    def _build_algorithm(self, obs_space, action_space):
        from tianshou.algorithm import SAC, TD3
        from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
        from tianshou.algorithm.modelfree.sac import SACPolicy
        from tianshou.algorithm.optim import AdamOptimizerFactory
        from tianshou.utils.net.common import Net
        from tianshou.utils.net.continuous import (
            ContinuousActorDeterministic,
            ContinuousActorProbabilistic,
            ContinuousCritic,
        )

        obs_shape = obs_space.shape
        action_shape = action_space.shape
        hidden_sizes = tuple(int(size) for size in self.config.hidden_sizes)
        if self.algorithm_name == "sac":
            policy_preprocess = Net(state_shape=obs_shape, hidden_sizes=hidden_sizes)
            policy_actor = ContinuousActorProbabilistic(
                preprocess_net=policy_preprocess,
                action_shape=action_shape,
                hidden_sizes=(),
                max_action=1.0,
            )
            policy = SACPolicy(
                actor=policy_actor,
                action_space=action_space,
                observation_space=obs_space,
                deterministic_eval=True,
            )
            return SAC(
                policy=policy,
                policy_optim=AdamOptimizerFactory(lr=self.config.policy_lr),
                critic=self._critic(Net, ContinuousCritic, obs_shape, action_shape, hidden_sizes),
                critic_optim=AdamOptimizerFactory(lr=self.config.critic_lr),
                critic2=self._critic(Net, ContinuousCritic, obs_shape, action_shape, hidden_sizes),
                critic2_optim=AdamOptimizerFactory(lr=self.config.critic_lr),
                tau=self.config.tau,
                gamma=self.config.gamma,
                alpha=0.2,
            )

        policy_preprocess = Net(state_shape=obs_shape, hidden_sizes=hidden_sizes)
        policy_actor = ContinuousActorDeterministic(
            preprocess_net=policy_preprocess,
            action_shape=action_shape,
            hidden_sizes=(),
            max_action=1.0,
        )
        policy = ContinuousDeterministicPolicy(
            actor=policy_actor,
            action_space=action_space,
            observation_space=obs_space,
            action_scaling=True,
        )
        return TD3(
            policy=policy,
            policy_optim=AdamOptimizerFactory(lr=self.config.policy_lr),
            critic=self._critic(Net, ContinuousCritic, obs_shape, action_shape, hidden_sizes),
            critic_optim=AdamOptimizerFactory(lr=self.config.critic_lr),
            critic2=self._critic(Net, ContinuousCritic, obs_shape, action_shape, hidden_sizes),
            critic2_optim=AdamOptimizerFactory(lr=self.config.critic_lr),
            tau=self.config.tau,
            gamma=self.config.gamma,
            policy_noise=self.config.policy_noise,
            update_actor_freq=self.config.update_actor_freq,
            noise_clip=self.config.noise_clip,
        )

    def _critic(self, net_cls, critic_cls, obs_shape, action_shape, hidden_sizes):
        return critic_cls(
            preprocess_net=net_cls(
                state_shape=obs_shape,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
            )
        )

    def act(self, obs, *, deterministic: bool = False) -> np.ndarray:
        from tianshou.data import Batch

        if self.policy is None or self.algorithm is None:
            raise RuntimeError("Trainer must be initialized before act()")
        if deterministic:
            self.algorithm.eval()
        else:
            self.algorithm.train()
        batch = Batch(obs=np.asarray(obs, dtype=np.float32).reshape(1, -1), info={})
        with torch.no_grad():
            policy_out = self.policy(batch)
        action = policy_out.act
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not deterministic and self.algorithm_name == "td3":
            action = action + np.random.normal(0.0, self.config.exploration_noise, size=action.shape)
        if self.action_low is not None and self.action_high is not None:
            action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float32)

    def collect(self, env, steps: int) -> CollectStats:
        from tianshou.data import Batch

        self.ensure_initialized(env.observation_space, env.action_space)
        obs, _ = env.reset()
        rewards: list[float] = []
        for _ in range(max(1, int(steps))):
            act = self.act(obs, deterministic=False)
            obs_next, rew, terminated, truncated, _ = env.step(act)
            self.replay.add(
                Batch(
                    obs=np.asarray(obs, dtype=np.float32),
                    act=np.asarray(act, dtype=np.float32),
                    rew=float(rew),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    done=bool(terminated or truncated),
                    obs_next=np.asarray(obs_next, dtype=np.float32),
                    info={},
                )
            )
            rewards.append(float(rew))
            obs = obs_next
            if terminated or truncated:
                obs, _ = env.reset()
        self.collect_steps += max(1, int(steps))
        self.last_reward_mean = float(np.mean(rewards)) if rewards else 0.0
        return CollectStats(steps=max(1, int(steps)), reward_mean=self.last_reward_mean)

    def update(self, gradient_steps: int) -> UpdateStats:
        if self.algorithm is None or self.policy is None or len(self.replay) == 0:
            return UpdateStats(gradient_steps=0, loss=0.0)
        from tianshou.algorithm.algorithm_base import policy_within_training_step

        steps = max(1, int(gradient_steps))
        last_loss = 0.0
        for _ in range(steps):
            sample_size = min(self.config.batch_size, len(self.replay))
            with policy_within_training_step(self.policy):
                stats = self.algorithm.update(self.replay, sample_size=sample_size)
            self.last_training_stats = self._stats_to_float_dict(stats)
            last_loss = self._loss_from_stats(self.last_training_stats)
            self.update_steps += 1
        self.last_loss = float(last_loss)
        return UpdateStats(gradient_steps=steps, loss=self.last_loss)

    def save(self, path: str) -> None:
        if self.algorithm is None:
            raise RuntimeError("Trainer must be initialized before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": self.algorithm.state_dict(), "diagnostics": self.diagnostics()}, path)

    def load(self, path: str) -> None:
        if self.algorithm is None:
            raise RuntimeError("Trainer must be initialized before load()")
        payload = torch.load(path, map_location="cpu")
        self.algorithm.load_state_dict(payload["algorithm"])

    def diagnostics(self) -> dict[str, float]:
        return {
            "trainer_algorithm_id": 1.0 if self.algorithm_name == "td3" else 0.0,
            "trainer_collect_steps": float(self.collect_steps),
            "trainer_update_steps": float(self.update_steps),
            "trainer_loss": float(self.last_loss),
            "trainer_reward_mean": float(self.last_reward_mean),
            "trainer_replay_size": float(len(self.replay)),
            **{f"trainer_{key}": value for key, value in self.last_training_stats.items()},
        }

    def _stats_to_float_dict(self, stats) -> dict[str, float]:
        raw = asdict(stats) if is_dataclass(stats) else getattr(stats, "__dict__", {})
        cleaned: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                cleaned[key] = float(value)
        return cleaned

    def _loss_from_stats(self, stats: dict[str, float]) -> float:
        losses = [
            abs(value)
            for key, value in stats.items()
            if key.endswith("loss") or key in {"actor_loss", "critic1_loss", "critic2_loss", "alpha_loss"}
        ]
        return float(np.sum(losses)) if losses else 0.0
