"""Gymnasium environment for TD3 backdoor policy training."""

from __future__ import annotations

from typing import Callable

import numpy as np

from fl_sandbox.attacks.rl_backdoor.attack import RLBackdoorAttack
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder
from fl_sandbox.attacks.rl_backdoor.reward import BackdoorRewardFn, BackdoorRewardInputs


try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - only used in minimal unit environments
    gym = None


class _Box:
    def __init__(self, *, low, high, shape=None, dtype=np.float32) -> None:
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = tuple(shape or self.low.shape)
        self.dtype = dtype


def _box(*, low, high, shape=None, dtype=np.float32):
    if gym is not None:
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _Box(low=low, high=high, shape=shape, dtype=dtype)


class BackdoorPolicyGymEnv:
    """Real-FL environment whose action is consumed by ``RLBackdoorAttack``."""

    def __init__(
        self,
        *,
        runner_factory: Callable[..., object],
        attack_factory: Callable[[], RLBackdoorAttack] | None = None,
        config: BackdoorRLConfig | None = None,
    ) -> None:
        self.runner_factory = runner_factory
        self.attack_factory = attack_factory or RLBackdoorAttack
        self.config = config or BackdoorRLConfig()
        self.reward_fn = BackdoorRewardFn(
            clean_weight=self.config.reward_clean_weight,
            norm_weight=self.config.reward_norm_weight,
        )
        self.observation_space = _box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = _box(
            low=self.config.action_low,
            high=self.config.action_high,
            dtype=np.float32,
        )
        self.runner = None
        self.attack = None
        self.observation_builder = BackdoorObservationBuilder(self.config)
        self.round_idx = 0
        self.last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.last_clean = 0.0
        self.last_asr = 0.0
        self.previous_weights = None

    def reset(self, *, seed: int | None = None, options=None):
        del options
        seed_offset = int(seed or 0)
        try:
            self.runner = self.runner_factory(seed_offset=seed_offset)
        except TypeError:
            self.runner = self.runner_factory(seed_offset)
        self.attack = self.attack_factory()
        self.round_idx = 0
        self.last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.last_clean = 0.0
        self.last_asr = 0.0
        self.previous_weights = [layer.copy() for layer in self.runner.current_weights]
        self.observation_builder.reset()
        return self._observe(), {"round_idx": self.round_idx}

    def step(self, action):
        if self.runner is None or self.attack is None:
            raise RuntimeError("BackdoorPolicyGymEnv must be reset before step()")
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[: self.config.action_dim], -1.0, 1.0)
        before_clean = float(self.last_clean)
        before_asr = float(self.last_asr)
        old_weights = [layer.copy() for layer in self.runner.current_weights]
        self.round_idx += 1
        summary = self.runner.run_round(
            self.round_idx,
            attack=self.attack,
            evaluate=True,
            attacker_action=action,
        )
        clean = _finite(getattr(summary, "clean_acc", np.nan), before_clean)
        asr = _finite(getattr(summary, "backdoor_acc", np.nan), before_asr)
        norm_ratio = _norm_ratio(summary)
        reward = self.reward_fn(
            BackdoorRewardInputs(
                asr_before=before_asr,
                asr_after=asr,
                clean_before=before_clean,
                clean_after=clean,
                norm_ratio=norm_ratio,
            )
        )
        self.previous_weights = old_weights
        self.last_action = action.copy()
        self.last_clean = clean
        self.last_asr = asr
        terminated = self.round_idx >= max(1, int(self.config.train_horizon))
        info = {
            "round_idx": self.round_idx,
            "clean_acc": clean,
            "asr": asr,
            "norm_ratio": norm_ratio,
        }
        return self._observe(), float(reward), bool(terminated), False, info

    def _observe(self) -> np.ndarray:
        sampled_attackers, sampled_clients = self._attacker_sampling_counts()
        return self.observation_builder.build(
            weights=self.runner.current_weights,
            previous_weights=self.previous_weights or self.runner.current_weights,
            last_action=self.last_action,
            round_idx=self.round_idx,
            total_rounds=max(1, int(self.config.train_horizon)),
            sampled_attacker_count=sampled_attackers,
            num_attackers=len(getattr(self.runner, "attacker_ids", []) or []),
            sampled_client_count=sampled_clients,
            clean_acc=self.last_clean,
            asr=self.last_asr,
        )

    def _attacker_sampling_counts(self) -> tuple[int, int]:
        attacker_ids = set(getattr(self.runner, "attacker_ids", []) or [])
        if not attacker_ids:
            return 0, 0
        sample_fn = getattr(self.runner, "_sample_clients", None)
        if sample_fn is None:
            return 0, 0
        sampled = set(sample_fn(max(1, self.round_idx + 1)))
        return len(attacker_ids.intersection(sampled)), len(sampled)


def _finite(value, fallback: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def _norm_ratio(summary) -> float:
    benign = getattr(summary, "benign_update_norms", []) or []
    malicious = getattr(summary, "malicious_update_norms", []) or []
    if not benign or not malicious:
        return 1.0
    benign_mean = float(np.mean(benign))
    if benign_mean <= 1e-12:
        return 1.0
    return float(np.mean(malicious)) / benign_mean
