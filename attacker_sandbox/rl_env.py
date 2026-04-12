"""Minimal single-agent gym-style RL environment for attacker sandbox experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:  # pragma: no cover - lightweight local fallback for sandbox use
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
            if seed is not None:
                np.random.seed(seed)
            return None

        def close(self) -> None:
            return None

    class Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _FallbackGymModule:
        Env = _FallbackEnv

    gym = _FallbackGymModule()

from src.models.cnn import MNISTClassifier, ResNet18, get_compressed_state

from .attacks import BFLAttack, BRLAttack, DBAAttack, IPMAttack, LMPAttack, RLAttack, SandboxAttack
from .fl_runner import MinimalFLRunner, RoundSummary, SandboxConfig


ATTACK_BUILDERS = {
    "ipm": lambda cfg: IPMAttack(scaling=cfg.ipm_scaling),
    "lmp": lambda cfg: LMPAttack(scale=cfg.lmp_scale),
    "bfl": lambda cfg: BFLAttack(poison_frac=cfg.bfl_poison_frac),
    "dba": lambda cfg: DBAAttack(
        num_sub_triggers=cfg.dba_num_sub_triggers,
        poison_frac=cfg.dba_poison_frac,
    ),
    "rl": lambda cfg: RLAttack(default_action=tuple(cfg.attacker_action)),
    "brl": lambda cfg: BRLAttack(default_action=tuple(cfg.attacker_action)),
}

BACKDOOR_ATTACKS = {"bfl", "dba", "brl"}


class AttackerRLEnv(gym.Env):
    """Single-agent RL wrapper around one attacker-controlled FL round per environment step."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        attack_type: str = "rl",
        max_rounds: int = 50,
        reward_mode: str = "auto",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or SandboxConfig()
        self.attack_type = attack_type.lower()
        if self.attack_type not in ATTACK_BUILDERS:
            raise ValueError(f"Unsupported attack_type: {attack_type}")
        self.max_rounds = max_rounds
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        obs_dim = self._infer_obs_dim()
        self.observation_space = Box(-1.0, 1.0, (obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (3,), dtype=np.float32)

        self.runner = MinimalFLRunner(self.config)
        self.attack = self._build_attack()
        self.round_idx = 0
        self.last_summary: Optional[RoundSummary] = None

    def _infer_obs_dim(self) -> int:
        if self.config.dataset == "cifar10":
            model = ResNet18()
        else:
            model = MNISTClassifier()
        state, _ = get_compressed_state(model, num_tail_layers=2)
        return int(state.shape[0])

    def _build_attack(self) -> SandboxAttack:
        attack_cfg = type(
            "AttackConfig",
            (),
            {
                "ipm_scaling": self.config.ipm_scaling if hasattr(self.config, "ipm_scaling") else 2.0,
                "lmp_scale": self.config.lmp_scale if hasattr(self.config, "lmp_scale") else 2.0,
                "bfl_poison_frac": self.config.bfl_poison_frac,
                "dba_poison_frac": self.config.dba_poison_frac,
                "dba_num_sub_triggers": self.config.dba_num_sub_triggers,
                "attacker_action": self.config.attacker_action,
            },
        )
        return ATTACK_BUILDERS[self.attack_type](attack_cfg)

    def _get_obs(self) -> np.ndarray:
        state, _ = get_compressed_state(self.runner.model, num_tail_layers=2)
        return state.astype(np.float32)

    def _compute_reward(self, summary: RoundSummary) -> float:
        if self.reward_mode == "clean_drop":
            return float(-summary.clean_acc)
        if self.reward_mode == "backdoor":
            return float(summary.backdoor_acc)
        if self.reward_mode == "balanced":
            return float(summary.backdoor_acc - summary.clean_acc)
        if self.attack_type in BACKDOOR_ATTACKS:
            return float(summary.backdoor_acc - summary.clean_acc)
        return float(-summary.clean_acc)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed
        if options and "attack_type" in options:
            self.attack_type = str(options["attack_type"]).lower()
            if self.attack_type not in ATTACK_BUILDERS:
                raise ValueError(f"Unsupported attack_type: {self.attack_type}")
            self.attack = self._build_attack()
        self.runner = MinimalFLRunner(self.config)
        self.round_idx = 0
        self.last_summary = None
        obs = self._get_obs()
        info = {
            "attack_type": self.attack_type,
            "config": asdict(self.config),
        }
        return obs, info

    def step(self, action: np.ndarray):
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.round_idx += 1
        summary = self.runner.run_round(
            self.round_idx,
            attack=self.attack,
            evaluate=True,
            attacker_action=clipped_action,
        )
        self.last_summary = summary
        obs = self._get_obs()
        reward = self._compute_reward(summary)
        terminated = self.round_idx >= self.max_rounds
        truncated = False
        info = {
            "round_idx": summary.round_idx,
            "attack_name": summary.attack_name,
            "clean_acc": summary.clean_acc,
            "clean_loss": summary.clean_loss,
            "backdoor_acc": summary.backdoor_acc,
            "mean_benign_norm": float(np.mean(summary.benign_update_norms)) if summary.benign_update_norms else 0.0,
            "mean_malicious_norm": float(np.mean(summary.malicious_update_norms)) if summary.malicious_update_norms else 0.0,
            "mean_malicious_cosine": float(np.mean(summary.malicious_cosines_to_benign))
            if summary.malicious_cosines_to_benign else 0.0,
            "selected_attackers": list(summary.selected_attackers),
            "sampled_clients": list(summary.sampled_clients),
            "action": clipped_action.tolist(),
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human" or self.last_summary is None:
            return
        print(
            f"round={self.last_summary.round_idx} "
            f"attack={self.last_summary.attack_name} "
            f"clean_acc={self.last_summary.clean_acc:.4f} "
            f"backdoor_acc={self.last_summary.backdoor_acc:.4f}"
        )

    def close(self) -> None:
        self.last_summary = None
