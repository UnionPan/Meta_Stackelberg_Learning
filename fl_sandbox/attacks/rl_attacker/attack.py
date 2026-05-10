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
from fl_sandbox.attacks.rl_attacker.observation import ProjectedObservationBuilder
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
class _PendingRealTransition:
    obs: np.ndarray
    action: np.ndarray
    previous_action: np.ndarray
    observation_builder: ProjectedObservationBuilder
    round_idx: int
    total_rounds: int
    defense_type: str


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
        self.observation_previous_weights: Weights | None = None
        self.last_aggregate_update: Weights | None = None
        self.last_action = np.zeros(self.config.action_dim("fedavg"), dtype=np.float32)
        self.observation_builder = ProjectedObservationBuilder(
            config=self.config,
            action_dim=self.config.action_dim("fedavg"),
        )
        self.ready = False
        self._last_policy_obs: np.ndarray | None = None
        self._pending_real_transition: _PendingRealTransition | None = None
        self._real_ppo_transition_count = 0
        self._real_ppo_buffered_steps = 0
        self._real_ppo_update_count = 0
        self._diagnostics = RLSim2RealDiagnostics(
            window=self.config.deploy_guard_window,
            max_gap=self.config.deploy_guard_max_sim2real_gap,
        )

    def observe_round(self, ctx) -> None:
        if torch is None or ctx.model is None or ctx.device is None:
            return
        if self.model_template is None:
            self.model_template = copy.deepcopy(ctx.model).cpu()
        self.device = ctx.device
        self.fl_config = ctx.fl_config
        self.defender = self._build_defender(ctx)
        if self.observation_builder.action_dim != self.config.action_dim(ctx.defense_type):
            self.observation_builder = ProjectedObservationBuilder(
                config=self.config,
                action_dim=self.config.action_dim(ctx.defense_type),
            )
        self.distribution_learner.initialize_from_loader(
            ctx.all_attacker_train_iter or ctx.attacker_train_iter, self.device
        )
        self.latest_policy_weights = [layer.copy() for layer in ctx.old_weights]
        previous_weights = self.prev_weights or ctx.old_weights
        self.observation_previous_weights = [layer.copy() for layer in previous_weights]
        self.last_aggregate_update = [current - previous for previous, current in zip(previous_weights, ctx.old_weights)]
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
        if self.ready and not self._is_ppo() and ctx.round_idx <= self.config.policy_train_end_round:
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

        gap = abs(self._diagnostics.gap_mean)
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

        previous_action = self.last_action.copy()
        action = self._resolve_policy_action(ctx, attacker_action)
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self._pending_real_transition = None
        if self._is_ppo() and attacker_action is None and self._last_policy_obs is not None:
            total_rounds = int(getattr(ctx.fl_config, "rounds", max(self.config.policy_train_end_round, ctx.round_idx, 1)) or 1)
            self._pending_real_transition = _PendingRealTransition(
                obs=self._last_policy_obs.copy(),
                action=self.last_action.copy(),
                previous_action=previous_action.copy(),
                observation_builder=copy.deepcopy(self.observation_builder),
                round_idx=int(ctx.round_idx),
                total_rounds=total_rounds,
                defense_type=str(ctx.defense_type),
            )
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
        real_reward, components = self._real_reward_from_feedback(**kwargs)
        self._diagnostics.record_gap(
            real_reward=float(real_reward or 0.0),
            simulated_reward=float(self._diagnostics.simulated_reward),
            components=components,
        )
        self._record_real_ppo_transition(real_reward=float(real_reward or 0.0), components=components, **kwargs)
        if self.trainer is not None:
            self._diagnostics.trainer = self.trainer.diagnostics()
        payload = self._diagnostics.as_dict()
        payload["rl_proxy_reconstruction_accept_rate"] = float(self.distribution_learner.buffer.reconstruction_accept_rate)
        payload["rl_proxy_mean_reconstruction_quality"] = float(self.distribution_learner.buffer.mean_reconstruction_quality)
        payload["rl_real_ppo_transitions"] = float(self._real_ppo_transition_count)
        payload["rl_real_ppo_buffered_steps"] = float(self._real_ppo_buffered_steps)
        payload["rl_real_ppo_updates"] = float(self._real_ppo_update_count)
        return payload

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
        self._last_policy_obs = None
        if attacker_action is not None:
            return np.asarray(attacker_action, dtype=np.float32)
        if self.trainer is not None:
            obs = self._current_observation(ctx)
            if self._is_ppo():
                self._ensure_real_trainer(ctx, obs)
            deterministic = not (self._is_ppo() and ctx.round_idx <= self.config.policy_train_end_round)
            action = np.asarray(self.trainer.act(obs, deterministic=deterministic), dtype=np.float32)
            self._last_policy_obs = obs.copy()
            return action
        if self._is_ppo():
            obs = self._current_observation(ctx)
            self._ensure_real_trainer(ctx, obs)
            deterministic = ctx.round_idx > self.config.policy_train_end_round
            action = np.asarray(self.trainer.act(obs, deterministic=deterministic), dtype=np.float32)
            self._last_policy_obs = obs.copy()
            return action
        low, high = self.config.action_bounds(ctx.defense_type)
        return (0.5 * (low + high)).astype(np.float32)

    def _current_observation(self, ctx) -> np.ndarray:
        if self.model_template is None or self.device is None:
            raise RuntimeError("RL attacker observation requested before initialization")
        total_rounds = int(getattr(ctx.fl_config, "rounds", max(self.config.policy_train_end_round, ctx.round_idx, 1)) or 1)
        return self.observation_builder.build(
            weights=ctx.old_weights,
            previous_weights=self.observation_previous_weights or ctx.old_weights,
            last_aggregate_update=self.last_aggregate_update or [np.zeros_like(layer) for layer in ctx.old_weights],
            last_action=self.last_action,
            last_bypass_score=float(self._diagnostics.deploy_guard_blocked),
            round_idx=ctx.round_idx,
            total_rounds=total_rounds,
            defense_type=ctx.defense_type,
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

    def _is_ppo(self) -> bool:
        return self.config.algorithm.lower() == "ppo"

    def _ensure_real_trainer(self, ctx, obs: np.ndarray) -> None:
        if self.trainer is None:
            self.trainer = build_trainer(self.config)
        if getattr(self.trainer, "initialized", False):
            return
        import gymnasium as gym

        low, high = self.config.action_bounds(ctx.defense_type)
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.asarray(obs, dtype=np.float32).shape, dtype=np.float32)
        action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.trainer.ensure_initialized(obs_space, action_space)

    def _real_reward_from_feedback(self, **kwargs) -> tuple[float, dict[str, float]]:
        real_reward = kwargs.get("real_reward")
        components: dict[str, float] = {}
        clean_loss_before = kwargs.get("clean_loss_before")
        clean_loss = kwargs.get("clean_loss")
        clean_acc_before = kwargs.get("clean_acc_before")
        clean_acc = kwargs.get("clean_acc")
        metric_values = (clean_loss_before, clean_loss, clean_acc_before, clean_acc)
        if all(value is not None and np.isfinite(float(value)) for value in metric_values):
            components["loss"] = float(clean_loss) - float(clean_loss_before)
            components["acc"] = float(clean_acc_before) - float(clean_acc)
            components["bypass"] = self._estimate_bypass_score(**kwargs)
            smoothness, saturation = self._real_action_penalties()
            components["smoothness"] = smoothness
            components["oob"] = saturation
            real_reward = (
                self.config.reward_loss_weight * components["loss"]
                + self.config.reward_accuracy_weight * components["acc"]
                + self.config.reward_bypass_weight * components["bypass"]
                - self.config.reward_action_smoothness_weight * components["smoothness"]
                - self.config.reward_action_saturation_weight * components["oob"]
            )
        elif real_reward is None:
            real_reward = kwargs.get("backdoor_acc", self._diagnostics.real_reward)
        return float(real_reward or 0.0), components

    def _real_action_penalties(self) -> tuple[float, float]:
        pending = self._pending_real_transition
        if pending is None:
            return 0.0, 0.0
        smoothness = float(np.sum((pending.action - pending.previous_action) ** 2))
        saturation = float(np.mean(np.isclose(np.abs(pending.action), 1.0)))
        return smoothness, saturation

    def _estimate_bypass_score(self, **kwargs) -> float:
        pending = self._pending_real_transition
        if pending is None:
            return 0.0
        defense = pending.defense_type.lower()
        if defense == "fedavg":
            return 0.0
        all_weights = kwargs.get("all_weights") or []
        malicious_indices = kwargs.get("malicious_indices") or []
        aggregated_weights = kwargs.get("aggregated_weights")
        if aggregated_weights is None or not all_weights or not malicious_indices:
            return 0.0
        aggregated_vec = self._weights_to_vector(aggregated_weights)
        malicious_vecs = [
            self._weights_to_vector(all_weights[idx])
            for idx in malicious_indices
            if 0 <= int(idx) < len(all_weights)
        ]
        if not malicious_vecs:
            return 0.0
        if defense in {"krum", "multi_krum"}:
            return float(any(np.allclose(aggregated_vec, vec, rtol=1e-5, atol=1e-7) for vec in malicious_vecs))
        distances = np.asarray([np.linalg.norm(aggregated_vec - vec) for vec in malicious_vecs], dtype=np.float64)
        scale = max(1e-8, float(np.linalg.norm(aggregated_vec)))
        return float(np.clip(1.0 - float(np.min(distances)) / scale, 0.0, 1.0))

    @staticmethod
    def _weights_to_vector(weights) -> np.ndarray:
        return np.concatenate([np.asarray(layer, dtype=np.float32).reshape(-1) for layer in weights]).astype(np.float32)

    def _record_real_ppo_transition(self, *, real_reward: float, components: dict[str, float], **kwargs) -> None:
        del components
        pending = self._pending_real_transition
        if not self._is_ppo() or pending is None or self.trainer is None:
            return
        if pending.round_idx > self.config.policy_train_end_round:
            self._pending_real_transition = None
            return
        ctx = kwargs.get("ctx")
        aggregated_weights = kwargs.get("aggregated_weights")
        if ctx is None or aggregated_weights is None:
            self._pending_real_transition = None
            return
        aggregate_update = [new - old for old, new in zip(ctx.old_weights, aggregated_weights)]
        next_builder = copy.deepcopy(pending.observation_builder)
        next_obs = next_builder.build(
            weights=aggregated_weights,
            previous_weights=ctx.old_weights,
            last_aggregate_update=aggregate_update,
            last_action=pending.action,
            last_bypass_score=float(self._estimate_bypass_score(**kwargs)),
            round_idx=pending.round_idx + 1,
            total_rounds=pending.total_rounds,
            defense_type=pending.defense_type,
        )
        self.trainer.add_transition(
            pending.obs,
            pending.action,
            reward=float(real_reward),
            obs_next=next_obs,
            terminated=False,
            truncated=False,
        )
        self._real_ppo_transition_count += 1
        self._real_ppo_buffered_steps += 1
        update_threshold = max(2, int(self.config.ppo_real_rollout_steps))
        if self._real_ppo_buffered_steps >= update_threshold:
            update_stats = self.trainer.update(gradient_steps=1)
            self._real_ppo_update_count += 1
            self._real_ppo_buffered_steps = 0
            self._diagnostics.trainer = {
                **self.trainer.diagnostics(),
                "trainer_last_update_loss": float(update_stats.loss),
            }
        self._pending_real_transition = None

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
