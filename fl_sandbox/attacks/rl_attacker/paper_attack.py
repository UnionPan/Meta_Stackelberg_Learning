"""Formal paper-aligned RL attacker using an offline Phase 1 distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch

from fl_sandbox.aggregators.rules import AggregationDefender
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.rl_attacker.config import PAPER_CLIPPED_MEDIAN, RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.observation import build_paper_clipped_median_observation
from fl_sandbox.attacks.rl_attacker.simulator.paper_env import (
    PaperAttackerPolicyGymEnv,
    PaperFLSimulator,
    craft_paper_malicious_update,
    decode_paper_action,
)
from fl_sandbox.attacks.rl_attacker.proxy.paper_dataset import PaperDistributionDataset, PaperDistributionSampler
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import update_norm
from fl_sandbox.attacks.rl_attacker.trainer import Trainer, build_trainer
from fl_sandbox.runtime import Weights


@dataclass
class PaperRLAttack(SandboxAttack):
    """Paper-style clipped-median attacker: offline TD3 warmup, frozen deployment."""

    default_action: tuple[float, float] = (0.0, 0.0)
    config: RLAttackerConfig = field(default_factory=RLAttackerConfig)

    name: str = "RL"
    attack_type: str = "rl"

    def __post_init__(self) -> None:
        if not self.config.distribution_dir:
            raise ValueError("attack_type='rl' requires rl_distribution_dir / --distribution_dir")
        if self.config.algorithm.lower() != "td3":
            raise ValueError("paper-aligned attack_type='rl' requires TD3")
        self.config.attacker_semantics = PAPER_CLIPPED_MEDIAN
        self.config.freeze_policy = True
        self.distribution = PaperDistributionDataset(
            self.config.distribution_dir,
            split=self.config.distribution_split,
        )
        self.sampler = PaperDistributionSampler(
            self.distribution,
            self.config.distribution_growth_mode,
            initial_samples=self.config.strict_reproduction_initial_samples,
            samples_per_episode=self.config.strict_reproduction_samples_per_epoch,
        )
        self.trainer: Optional[Trainer] = None
        self._policy_warmup_done = False
        self._policy_random_warmup_done = False
        self._policy_training_frozen = False
        self._policy_train_steps_completed = 0
        self._ready = len(self.distribution) > 0
        self._last_ctx = None
        self._last_action = np.asarray(self.default_action, dtype=np.float32)
        self._last_simulated_reward = 0.0
        self._last_real_reward = 0.0
        self._last_action_metrics: dict[str, float] = {}
        self._last_observation_metrics: dict[str, float] = {}
        self._policy_initial_weights: Optional[Weights] = None
        self._saved_policy_warmup_steps: set[int] = set()

    def observe_round(self, ctx) -> None:
        self._last_ctx = ctx
        self.config.validate_defense(ctx.defense_type)
        self._remember_policy_initial_weights(ctx)
        if not self._policy_warmup_done:
            checkpoint_path = self.config.policy_checkpoint_for_round(int(ctx.round_idx))
            if checkpoint_path:
                self._load_policy_checkpoint(ctx, checkpoint_path)
            else:
                self._train_policy_offline(ctx)

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> list[Weights]:
        del attacker_action
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers <= 0:
            return []
        if not self._ready or ctx.model is None or ctx.device is None:
            return self.fallback_old_weights(ctx)
        if int(ctx.round_idx) < int(self.config.attack_start_round or 0):
            return self.fallback_old_weights(ctx)
        if not self._policy_warmup_done:
            return self.fallback_old_weights(ctx)

        obs = self._current_observation(ctx)
        action = self._policy_action(obs)
        gamma, local_steps = decode_paper_action(action)
        self._last_action = action
        self._last_action_metrics = {
            "rl_action_gamma": float(gamma),
            "rl_action_local_steps": float(local_steps),
            "rl_action_dim": float(action.size),
            "rl_action_norm": float(np.linalg.norm(action)),
        }
        crafted = craft_paper_malicious_update(
            model_template=ctx.model,
            old_weights=ctx.old_weights,
            distribution=self.sampler,
            device=ctx.device,
            lr=float(self.config.paper_local_lr),
            local_steps=local_steps,
            gamma=gamma,
            batch_size=int(self.config.local_search_batch_size),
        )
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]

    def after_round(self, **kwargs) -> dict[str, float]:
        before = kwargs.get("clean_loss_before", None)
        after = kwargs.get("clean_loss", None)
        if before is not None and after is not None:
            self._last_real_reward = float(after) - float(before)
        gap = abs(float(self._last_real_reward) - float(self._last_simulated_reward))
        payload = {
            "rl_proxy_source": 1.0,
            "rl_proxy_buffer_size": float(len(self.distribution)),
            "rl_policy_warmup_done": float(self._policy_warmup_done),
            "rl_policy_frozen": float(self._policy_training_frozen),
            "rl_policy_train_steps_completed": float(self._policy_train_steps_completed),
            "rl_simulated_reward": float(self._last_simulated_reward),
            "rl_real_reward": float(self._last_real_reward),
            "rl_sim2real_gap": float(gap),
        }
        if self.trainer is not None:
            payload.update(
                {
                    f"rl_{key}": value
                    for key, value in self.trainer.diagnostics().items()
                }
            )
        payload.update(self._last_action_metrics)
        payload.update(self._last_observation_metrics)
        ctx = kwargs.get("ctx", None)
        if ctx is not None and getattr(ctx, "benign_weights", None):
            payload["rl_real_benign_update_norm_mean"] = float(
                np.mean([update_norm(ctx.old_weights, weights) for weights in ctx.benign_weights])
            )
        return payload

    def _train_policy_offline(self, ctx) -> None:
        env = self._build_policy_env(ctx)
        self.trainer = self.trainer or build_trainer(self.config)
        self.trainer.ensure_initialized(env.observation_space, env.action_space)
        started = time.perf_counter()
        total_steps = max(0, int(self.config.policy_warmup_steps))
        train_freq = max(1, int(self.config.train_freq_steps or 1))
        print(
            f"RL paper TD3 offline training starting: total_steps={total_steps}, "
            f"random_steps={int(self.config.policy_warmup_random_steps)}, "
            f"train_freq={train_freq}, horizon={int(self.config.simulator_horizon)}",
            flush=True,
        )

        if not self._policy_random_warmup_done:
            random_steps = min(
                max(0, int(self.config.policy_warmup_random_steps)),
                total_steps,
            )
            if random_steps > 0:
                stats = self._trainer_warmup_collect(env, random_steps=random_steps, reset_on_start=True)
                self._last_simulated_reward = float(stats.reward_mean)
                self._policy_train_steps_completed += int(stats.steps)
                self._log_warmup_progress("random", self._policy_train_steps_completed, total_steps, started)
            self._policy_random_warmup_done = True

        while self._policy_train_steps_completed < total_steps:
            remaining = total_steps - self._policy_train_steps_completed
            steps = min(train_freq, remaining)
            collect_stats = self._trainer_collect(env, steps=steps, reset_on_start=False)
            self.trainer.update(gradient_steps=steps)
            self._last_simulated_reward = float(collect_stats.reward_mean)
            self._policy_train_steps_completed += int(collect_stats.steps)
            if (
                self._policy_train_steps_completed == total_steps
                or self._policy_train_steps_completed % max(1000, train_freq) == 0
            ):
                self._log_warmup_progress("td3", self._policy_train_steps_completed, total_steps, started)
            self._maybe_save_policy_warmup_checkpoint(self._policy_train_steps_completed)
        self._policy_warmup_done = True
        self._policy_training_frozen = True
        self._maybe_save_policy_warmup_checkpoint(self._policy_train_steps_completed, force=True)
        self._log_warmup_progress("done", self._policy_train_steps_completed, total_steps, started)

    def _maybe_save_policy_warmup_checkpoint(self, completed_steps: int, *, force: bool = False) -> None:
        if self.trainer is None or not hasattr(self.trainer, "save"):
            return
        directory = str(getattr(self.config, "policy_warmup_checkpoint_dir", "") or "")
        if not directory:
            return
        interval = max(0, int(getattr(self.config, "policy_warmup_checkpoint_interval", 0) or 0))
        completed_steps = max(0, int(completed_steps))
        if completed_steps <= 0:
            return
        should_save = bool(force)
        if interval > 0 and completed_steps % interval == 0:
            should_save = True
        if not should_save or completed_steps in self._saved_policy_warmup_steps:
            return
        path = Path(directory) / f"rl_policy_step_{completed_steps:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save(str(path))
        self._saved_policy_warmup_steps.add(completed_steps)
        print(f"RL paper TD3 warmup checkpoint saved: {path}", flush=True)

    def _trainer_warmup_collect(self, env, *, random_steps: int, reset_on_start: bool):
        try:
            return self.trainer.warmup_collect(env, random_steps=random_steps, reset_on_start=reset_on_start)
        except TypeError:
            return self.trainer.warmup_collect(env, random_steps=random_steps)

    def _trainer_collect(self, env, *, steps: int, reset_on_start: bool):
        try:
            return self.trainer.collect(env, steps=steps, reset_on_start=reset_on_start)
        except TypeError:
            return self.trainer.collect(env, steps=steps)

    def _build_policy_env(self, ctx) -> PaperAttackerPolicyGymEnv:
        self._remember_policy_initial_weights(ctx)
        defender = self._build_defender(ctx)
        simulator = PaperFLSimulator(
            model_template=ctx.model,
            distribution=self.sampler,
            defender=defender,
            config=self.config,
            fl_config=ctx.fl_config,
            device=ctx.device,
            eval_loader=None,
        )
        return PaperAttackerPolicyGymEnv(simulator, self.config, ctx.defense_type, self._policy_initial_weights)

    def _remember_policy_initial_weights(self, ctx) -> None:
        if self._policy_initial_weights is None:
            self._policy_initial_weights = [layer.copy() for layer in ctx.old_weights]

    def _load_policy_checkpoint(self, ctx, checkpoint_path: str) -> None:
        self.trainer = self.trainer or build_trainer(self.config)
        obs_space, action_space = self._deployment_spaces(ctx)
        self.trainer.ensure_initialized(obs_space, action_space)
        self.trainer.load(str(checkpoint_path))
        self._policy_warmup_done = True
        self._policy_random_warmup_done = True
        self._policy_training_frozen = True
        print(f"RL paper policy checkpoint loaded: {checkpoint_path}; offline simulator warmup skipped", flush=True)

    def _deployment_spaces(self, ctx):
        import gymnasium as gym

        obs = np.asarray(self._current_observation(ctx), dtype=np.float32)
        low, high = self.config.action_bounds(ctx.defense_type)
        return (
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32),
            gym.spaces.Box(low=low, high=high, dtype=np.float32),
        )

    def _log_warmup_progress(self, phase: str, completed: int, total: int, started: float) -> None:
        total = max(1, int(total))
        completed = max(0, int(completed))
        elapsed = max(1e-6, time.perf_counter() - started)
        rate = completed / elapsed
        remaining = max(0, total - completed)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"RL paper TD3 warmup {phase}: {completed}/{total} "
            f"steps ({completed / total:.1%}), elapsed={elapsed:.1f}s, eta={eta:.1f}s",
            flush=True,
        )

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        if self.trainer is None:
            return np.asarray(self.default_action, dtype=np.float32)
        return np.asarray(self.trainer.act(obs, deterministic=True), dtype=np.float32)

    def _current_observation(self, ctx) -> np.ndarray:
        obs = build_paper_clipped_median_observation(
            ctx.old_weights,
            num_attackers=max(1, self.selected_attacker_count(ctx)),
            tail_layers=self.config.state_tail_layers,
        )
        self._last_observation_metrics = self._observation_metrics(obs)
        return obs

    @staticmethod
    def _observation_metrics(obs: np.ndarray) -> dict[str, float]:
        values = np.asarray(obs, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return {"rl_observation_dim": 0.0}
        return {
            "rl_observation_dim": float(values.size),
            "rl_observation_norm": float(np.linalg.norm(values)),
            "rl_observation_mean": float(np.mean(values)),
            "rl_observation_std": float(np.std(values)),
            "rl_observation_min": float(np.min(values)),
            "rl_observation_max": float(np.max(values)),
        }

    @staticmethod
    def _build_defender(ctx) -> AggregationDefender:
        defender_cfg = getattr(getattr(ctx, "fl_config", None), "defender", None)
        return AggregationDefender(
            defense_type=str(getattr(ctx, "defense_type", "clipped_median")),
            krum_attackers=int(getattr(defender_cfg, "krum_attackers", 1) or 1),
            multi_krum_selected=getattr(defender_cfg, "multi_krum_selected", None),
            clipped_median_norm=float(getattr(defender_cfg, "clipped_median_norm", 2.0) or 2.0),
            trimmed_mean_ratio=float(getattr(defender_cfg, "trimmed_mean_ratio", 0.2) or 0.2),
            geometric_median_iters=int(getattr(defender_cfg, "geometric_median_iters", 10) or 10),
        )


RLAttack = PaperRLAttack
