"""Public adaptive RL attacker powered by the split Tianshou package."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from fl_sandbox.aggregators.rules import AggregationDefender
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.diagnostics import RLSim2RealDiagnostics, deploy_guard_allows
from fl_sandbox.attacks.rl_attacker.observation import build_observation_from_state
from fl_sandbox.attacks.rl_attacker.proxy import GradientDistributionLearner
from fl_sandbox.attacks.rl_attacker.simulator import AttackerPolicyGymEnv, SimulatedFLEnv
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import (
    build_model_from_template,
    craft_malicious_update,
    match_update_norm,
    update_norm,
)
from fl_sandbox.attacks.rl_attacker.trainer import Trainer, build_trainer
from fl_sandbox.core.runtime import Weights


@dataclass
class RLAttack(SandboxAttack):
    """Paper-style RL attacker with online proxy learning and Tianshou training."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    config: Optional[RLAttackerConfig] = None
    name: str = "RL"
    attack_type: str = "rl"

    def __post_init__(self) -> None:
        self.config = self.config or RLAttackerConfig()
        self.distribution_learner = GradientDistributionLearner(self.config)
        self.trainer: Trainer | None = None
        self.model_template = None
        self.device = torch.device("cpu") if torch is not None else None
        self.fl_config = None
        self.defender: AggregationDefender | None = None
        self.prev_weights: Weights | None = None
        self.prev_round_idx: int | None = None
        self.latest_policy_weights: Weights | None = None
        self.ready = False
        self._diagnostics = RLSim2RealDiagnostics()

    def observe_round(self, ctx) -> None:
        if torch is None or ctx.model is None or ctx.device is None:
            return
        if self.model_template is None:
            self.model_template = copy.deepcopy(ctx.model).cpu()
        self.device = ctx.device
        self.fl_config = ctx.fl_config
        self.defender = self._build_defender(ctx)
        self.distribution_learner.initialize_from_loader(
            ctx.all_attacker_train_iter or ctx.attacker_train_iter, self.device
        )
        self.latest_policy_weights = [layer.copy() for layer in ctx.old_weights]
        if (
            self.prev_weights is not None
            and ctx.round_idx <= self.config.distribution_steps
            and self.model_template is not None
        ):
            round_gap = ctx.round_idx - (self.prev_round_idx or ctx.round_idx - 1)
            self.distribution_learner.observe_server_update(
                previous_weights=self.prev_weights,
                current_weights=ctx.old_weights,
                model_template=self.model_template,
                server_lr=max(1e-8, float(ctx.server_lr)),
                round_gap=round_gap,
            )
        self.prev_weights = [layer.copy() for layer in ctx.old_weights]
        self.prev_round_idx = ctx.round_idx
        self.ready = (
            len(self.distribution_learner.buffer) > 0
            and self.model_template is not None
            and self.defender is not None
        )
        if self.ready and ctx.round_idx <= self.config.policy_train_end_round:
            self._train_policy()

    def execute(self, ctx, attacker_action=None):
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0:
            return []
        if (
            not self.ready
            or ctx.round_idx < self.config.attack_start_round
            or self.model_template is None
            or self.device is None
        ):
            return self._fallback_benign_weights(ctx)

        gap = abs(self._diagnostics.simulated_reward - self._diagnostics.real_reward)
        allowed = deploy_guard_allows(
            proxy_samples=len(self.distribution_learner.buffer),
            sim2real_gap=gap,
            min_proxy_samples=self.config.deploy_guard_min_proxy_samples,
            max_gap=self.config.deploy_guard_max_sim2real_gap,
        )
        self._diagnostics.gap = gap
        self._diagnostics.deploy_guard_blocked = not allowed
        if not allowed:
            return self._fallback_benign_weights(ctx)

        action = self._resolve_policy_action(ctx, attacker_action)
        decoded = decode_action(action, ctx.defense_type, self.config)
        crafted = craft_malicious_update(
            model_template=self.model_template,
            old_weights=ctx.old_weights,
            proxy_buffer=self.distribution_learner.buffer,
            device=self.device,
            params=decoded,
            search_batch_size=self.config.local_search_batch_size,
            state_tail_layers=self.config.state_tail_layers,
        )
        benign_norm_mean = (
            float(np.mean([update_norm(ctx.old_weights, weights) for weights in ctx.benign_weights]))
            if ctx.benign_weights
            else 0.0
        )
        if benign_norm_mean > 0:
            crafted = match_update_norm(
                ctx.old_weights,
                crafted,
                target_norm=benign_norm_mean * self.config.target_norm_ratio(ctx.defense_type),
            )
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]

    def after_round(self, **kwargs):
        real_reward = kwargs.get("real_reward")
        if real_reward is None:
            real_reward = kwargs.get("backdoor_acc", self._diagnostics.real_reward)
        self._diagnostics.real_reward = float(real_reward or 0.0)
        self._diagnostics.gap = abs(self._diagnostics.simulated_reward - self._diagnostics.real_reward)
        if self.trainer is not None:
            self._diagnostics.trainer = self.trainer.diagnostics()
        return self._diagnostics.as_dict()

    def _fallback_benign_weights(self, ctx) -> list[Weights]:
        from fl_sandbox.attacks.base import train_on_loader

        fallback_weights: list[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            loader = (ctx.selected_attacker_train_loaders or {}).get(attacker_id)
            if loader is None:
                fallback_weights.append([layer.copy() for layer in ctx.old_weights])
                continue
            fallback_weights.append(train_on_loader(ctx, loader))
        return fallback_weights

    def _resolve_policy_action(self, ctx, attacker_action) -> np.ndarray:
        if attacker_action is not None:
            return np.asarray(attacker_action, dtype=np.float32)
        if self.trainer is not None:
            return np.asarray(self.trainer.act(self._current_observation(ctx), deterministic=True), dtype=np.float32)
        low, high = self.config.action_bounds(ctx.defense_type)
        return (0.5 * (low + high)).astype(np.float32)

    def _current_observation(self, ctx) -> np.ndarray:
        if self.model_template is None or self.device is None:
            raise RuntimeError("RL attacker observation requested before initialization")
        from fl_sandbox.models import get_compressed_state

        model = build_model_from_template(self.model_template, ctx.old_weights, self.device)
        compressed, _ = get_compressed_state(model, num_tail_layers=self.config.state_tail_layers)
        return build_observation_from_state(
            {"pram": compressed.astype(np.float32), "num_attacker": len(ctx.selected_attacker_ids)},
            self._max_attackers(ctx),
        )

    def _max_attackers(self, ctx) -> int:
        fl_config = ctx.fl_config or self.fl_config
        num_clients = max(1, int(getattr(fl_config, "num_clients", 1) or 1))
        num_attackers = max(1, int(getattr(fl_config, "num_attackers", 1) or 1))
        subsample_rate = float(getattr(fl_config, "subsample_rate", 1.0) or 1.0)
        sampled_clients = max(1, int(num_clients * subsample_rate))
        return max(1, min(num_attackers, sampled_clients))

    def _build_defender(self, ctx) -> AggregationDefender:
        fl_config = ctx.fl_config
        return AggregationDefender(
            defense_type=ctx.defense_type,
            krum_attackers=getattr(fl_config, "krum_attackers", 1),
            multi_krum_selected=getattr(fl_config, "multi_krum_selected", None),
            clipped_median_norm=getattr(fl_config, "clipped_median_norm", 2.0),
            trimmed_mean_ratio=getattr(fl_config, "trimmed_mean_ratio", 0.2),
            geometric_median_iters=getattr(fl_config, "geometric_median_iters", 10),
        )

    def _train_policy(self) -> None:
        if (
            self.model_template is None
            or self.defender is None
            or self.fl_config is None
            or self.device is None
            or self.latest_policy_weights is None
        ):
            return
        if len(self.distribution_learner.buffer) < max(1, self.config.reconstruction_batch_size):
            return

        simulator = SimulatedFLEnv(
            model_template=self.model_template,
            proxy_buffer=self.distribution_learner.buffer,
            defender=self.defender,
            config=self.config,
            fl_config=self.fl_config,
            device=self.device,
        )
        env = AttackerPolicyGymEnv(
            simulator=simulator,
            rl_config=self.config,
            defense_type=self.defender.defense_type,
            initial_weights=self.latest_policy_weights,
        )
        if self.trainer is None:
            self.trainer = build_trainer(self.config)
        steps = max(1, self.config.episodes_per_observation) * max(1, self.config.simulator_horizon)
        collect_stats = self.trainer.collect(env, steps=steps)
        update_stats = self.trainer.update(gradient_steps=max(1, steps // max(1, self.config.train_freq_steps)))
        self._diagnostics.simulated_reward = collect_stats.reward_mean
        self._diagnostics.trainer = {
            **self.trainer.diagnostics(),
            "trainer_last_update_loss": float(update_stats.loss),
        }
