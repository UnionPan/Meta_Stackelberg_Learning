"""Policy implementations for RL backdoor attacks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_attacker.trainer import build_trainer


class EliteBackdoorPolicy:
    """Legacy baseline that keeps the best observed action by scalar reward."""

    def __init__(self, default_action=(1.0, 0.0, -1.0, 0.0)) -> None:
        self.default_action = np.asarray(default_action, dtype=np.float32)
        self.best_action = self.default_action.copy()
        self.best_reward = float("-inf")

    def act(self) -> np.ndarray:
        return self.best_action.copy()

    def observe(self, *, action, reward: float) -> None:
        reward = float(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = np.asarray(action, dtype=np.float32).reshape(-1)[:4].copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_action": [float(value) for value in self.best_action],
            "best_reward": float(self.best_reward),
            "default_action": [float(value) for value in self.default_action],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EliteBackdoorPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(default_action=payload.get("default_action", (1.0, 0.0, -1.0, 0.0)))
        policy.best_action = np.asarray(payload["best_action"], dtype=np.float32)
        policy.best_reward = float(payload["best_reward"])
        return policy


@dataclass
class BackdoorPolicyTrainStats:
    collect: object
    update: object


class TD3BackdoorPolicy:
    """TD3 actor/critic policy for paper-style backdoor action selection."""

    def __init__(self, config: BackdoorRLConfig | None = None, trainer_factory=build_trainer) -> None:
        self.config = config or BackdoorRLConfig()
        self.trainer_factory = trainer_factory
        self.trainer = self.trainer_factory(self.config)
        self.initialized = False

    def ensure_initialized(self, observation_space, action_space) -> None:
        self.trainer.ensure_initialized(observation_space, action_space)
        self.initialized = True

    def train(self, env, *, steps: int | None = None, gradient_steps: int | None = None) -> BackdoorPolicyTrainStats:
        self.ensure_initialized(env.observation_space, env.action_space)
        collect_steps = max(1, int(steps if steps is not None else self.config.train_steps))
        collect = self.trainer.collect(env, collect_steps)
        update_steps = max(
            1,
            int(gradient_steps if gradient_steps is not None else self.config.policy_train_gradient_steps()),
        )
        update = self.trainer.update(update_steps)
        return BackdoorPolicyTrainStats(collect=collect, update=update)

    def act(self, obs, *, deterministic: bool = True) -> np.ndarray:
        if not self.initialized and getattr(self.trainer, "policy", None) is None:
            raise RuntimeError("TD3BackdoorPolicy must be initialized before act()")
        action = self.trainer.act(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[: self.config.action_dim], -1.0, 1.0)

    def save(self, path: str | Path) -> None:
        self.trainer.save(str(path))

    def load(self, path: str | Path, observation_space, action_space) -> None:
        self.ensure_initialized(observation_space, action_space)
        self.trainer.load(str(path))

    def diagnostics(self) -> dict[str, float]:
        return self.trainer.diagnostics()
