"""Per-attack task execution for Meta-SG pre-training."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Callable, Dict, List, Optional

import numpy as np

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.games.trajectory import Trajectory
from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.attacks.base import AttackStrategy
from meta_sg.strategies.attacks.fixed import build_fixed_attack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackDecision, AttackType, DefenseDecision
from src.defenses import apply_post_defense


class NativeSandboxAttackMarker:
    """Name-only marker that makes FLSandboxCoordinatorAdapter build native fl_sandbox attacks."""

    def __init__(self, attack_type: AttackType) -> None:
        self.attack_type = attack_type
        self.name = attack_type.name


@dataclass
class TaskResult:
    attack_type: AttackType
    adapted_params: Dict
    mean_defender_reward: float
    mean_attacker_reward: float
    defender_reward_sum: float
    trajectories_collected: int
    transitions_collected: int
    # TD3 training signal for the local defender (mean over all gradient steps)
    defender_losses: Dict[str, float] = field(default_factory=dict)
    # Best-response losses for adaptive attackers (empty for non-adaptive)
    attacker_br_losses: Dict[str, float] = field(default_factory=dict)
    # ||θ_ξ(final) - θ_meta|| — per-task inner adaptation magnitude
    inner_delta_norm: float = 0.0
    # Held-out query evaluation for objectives that train for post-adaptation improvement.
    query_base_reward: float = float("nan")
    query_adapted_reward: float = float("nan")
    query_gain: float = float("nan")
    query_base_clean_acc: float = float("nan")
    query_adapted_clean_acc: float = float("nan")
    query_clean_drop: float = float("nan")
    query_base_backdoor_acc: float = float("nan")
    query_adapted_backdoor_acc: float = float("nan")
    query_gain_accepted: bool = True
    query_clean_accepted: bool = True
    query_backdoor_accepted: bool = True
    query_accepted: bool = True
    # Env-level diagnostics (action distribution, clean/backdoor acc)
    diagnostics: dict = field(default_factory=dict)


class AttackTaskRunner:
    """Runs one sampled attack task ξ from a common meta-policy start."""

    def __init__(
        self,
        coordinator_factory: Callable[..., FLCoordinator],
        td3_config: TD3Config,
        meta_config: MetaSGConfig,
        obs_dim: int,
        act_dim: int,
        attacker_agents: Dict[str, TD3Agent],
        attacker_buffers: Dict[str, ReplayBuffer],
        best_response: AttackerBestResponse,
        attacker_act_dim: int = 3,
    ) -> None:
        self.coordinator_factory = coordinator_factory
        self.td3_config = td3_config
        self.meta_config = meta_config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.attacker_act_dim = attacker_act_dim
        self.attacker_agents = attacker_agents
        self.attacker_buffers = attacker_buffers
        self.best_response = best_response

    def run(self, attack_type: AttackType, meta_defender: TD3Agent, seed_base: int) -> TaskResult:
        local_defender = meta_defender.clone()
        local_def_buffer = ReplayBuffer(
            self.td3_config.buffer_capacity, self.obs_dim, self.act_dim
        )

        env = self._build_env(attack_type, horizon=self.meta_config.H, seed=seed_base)

        collector = TrajectoryCollector(
            env=env,
            defender=local_defender,
            attacker=self._rollout_attacker_policy(attack_type),
            defender_buffer=local_def_buffer,
            attacker_buffer=self.attacker_buffers.get(attack_type.name),
            exploration_noise=self.td3_config.exploration_noise,
            store_attacker=attack_type.adaptive,
        )

        warmup_steps = self._warmup_steps()
        if warmup_steps > 0:
            collector.warmup(warmup_steps)

        reward_sum_D = 0.0
        reward_sum_A = 0.0
        trajectory_count = 0
        transition_count = 0

        all_def_losses: List[Dict] = []
        support_update_calls = 0
        support_episodes = max(1, int(getattr(self.meta_config, "support_episodes", 1)))

        action_diag: dict = {}
        attacker_action_diag: dict = {}
        env_diag: dict = {}
        for episode in range(support_episodes):
            # Phase 1: collect one support trajectory, then l TD3 gradient updates.
            traj = collector.collect(
                self.meta_config.H,
                seed=seed_base + episode * 1_000,
            )
            if trajectory_count == 0:
                action_diag = _action_diagnostics(traj)
                attacker_action_diag = _attacker_action_diagnostics(traj)
                env_diag = _env_diagnostics(traj)
            else:
                action_diag = _merge_weighted(
                    action_diag,
                    _action_diagnostics(traj),
                    trajectory_count,
                    1,
                )
                attacker_action_diag = _merge_weighted(
                    attacker_action_diag,
                    _attacker_action_diagnostics(traj),
                    trajectory_count,
                    1,
                )
                env_diag = _merge_weighted(env_diag, _env_diagnostics(traj), trajectory_count, 1)
            reward_sum_D += sum(t.defender_reward for t in traj.transitions)
            reward_sum_A += sum(t.attacker_reward for t in traj.transitions)
            transition_count += len(traj.transitions)
            trajectory_count += 1

            for _ in range(self.meta_config.l):
                support_update_calls += 1
                step_losses = local_defender.update(local_def_buffer)
                if step_losses:
                    all_def_losses.append(step_losses)

        # Phase 2 (adaptive only): attacker best-response, then optional defender update.
        attacker_br_losses: Dict[str, float] = {}
        if attack_type.adaptive:
            attacker_br_losses = self.best_response.update(attack_type)
            # attack_strategy.agent and collector.attacker are the same shared
            # object as attacker_agents[xi.name]; best_response updated it in-place.

            if self.meta_config.post_br_defender_updates > 0:
                traj2 = collector.collect(self.meta_config.H, seed=seed_base + 10_000)
                action_diag = _merge_weighted(action_diag, _action_diagnostics(traj2),
                                              trajectory_count, 1)
                attacker_action_diag = _merge_weighted(
                    attacker_action_diag,
                    _attacker_action_diagnostics(traj2),
                    trajectory_count,
                    1,
                )
                env_diag    = _merge_weighted(env_diag, _env_diagnostics(traj2),
                                              trajectory_count, 1)
                reward_sum_D += sum(t.defender_reward for t in traj2.transitions)
                reward_sum_A += sum(t.attacker_reward for t in traj2.transitions)
                transition_count += len(traj2.transitions)
                trajectory_count += 1

            for _ in range(self.meta_config.post_br_defender_updates):
                step_losses = local_defender.update(local_def_buffer)
                if step_losses:
                    all_def_losses.append(step_losses)

        # Inner adaptation magnitude: ||θ_ξ(final) - θ_meta||
        meta_p    = meta_defender.get_params()
        adapted_p = local_defender.get_params()
        inner_delta_norm = math.sqrt(sum(
            float((adapted_p[k] - meta_p[k]).norm() ** 2)
            for k in adapted_p
        ))

        diagnostics = {
            "buffer_size": len(local_def_buffer),
            "attacker_buffer_size": len(self.attacker_buffers[attack_type.name])
            if attack_type.name in self.attacker_buffers
            else 0,
            "support_episodes": support_episodes,
            "support_update_calls": support_update_calls,
            **action_diag,
            **attacker_action_diag,
            **env_diag,
        }
        query_base_reward = float("nan")
        query_adapted_reward = float("nan")
        query_gain = float("nan")
        query_base_clean_acc = float("nan")
        query_adapted_clean_acc = float("nan")
        query_clean_drop = float("nan")
        query_base_backdoor_acc = float("nan")
        query_adapted_backdoor_acc = float("nan")
        query_gain_accepted = True
        query_clean_accepted = True
        query_backdoor_accepted = True
        query_accepted = True
        diagnostics_only = False
        if (
            self.meta_config.meta_objective in {"query_gated_reptile", "query_targeted_reptile"}
            or self.meta_config.query_diagnostics_horizon is not None
        ):
            diagnostics_only = self.meta_config.meta_objective == "reptile"
            query_horizon = (
                self.meta_config.query_diagnostics_horizon
                or self.meta_config.query_horizon
                or self.meta_config.H
            )
            query_seed = seed_base + self.meta_config.query_seed_offset
            query_base = self._query_evaluate(
                attack_type=attack_type,
                defender=meta_defender,
                horizon=query_horizon,
                seed=query_seed,
            )
            query_adapted = self._query_evaluate(
                attack_type=attack_type,
                defender=local_defender,
                horizon=query_horizon,
                seed=query_seed,
            )
            query_base_reward = query_base["reward"]
            query_adapted_reward = query_adapted["reward"]
            query_gain = query_adapted_reward - query_base_reward
            query_base_clean_acc = query_base["clean_acc"]
            query_adapted_clean_acc = query_adapted["clean_acc"]
            query_clean_drop = _clean_drop(query_base_clean_acc, query_adapted_clean_acc)
            query_base_backdoor_acc = query_base["backdoor_acc"]
            query_adapted_backdoor_acc = query_adapted["backdoor_acc"]
            query_gain_accepted = query_gain >= self.meta_config.query_accept_margin
            query_clean_accepted = _query_clean_constraints_accept(
                base_clean=query_base_clean_acc,
                adapted_clean=query_adapted_clean_acc,
                clean_floor=self.meta_config.query_clean_floor,
                clean_drop_tolerance=self.meta_config.query_clean_drop_tolerance,
            )
            if self.meta_config.meta_objective == "query_targeted_reptile":
                if str(attack_type.objective) == "targeted":
                    query_backdoor_accepted = _query_backdoor_reduction_accept(
                        base_backdoor=query_base_backdoor_acc,
                        adapted_backdoor=query_adapted_backdoor_acc,
                        reduction_margin=self.meta_config.query_targeted_asr_reduction_margin,
                        min_base_backdoor=self.meta_config.query_targeted_min_base_backdoor,
                    )
                    query_accepted = query_clean_accepted and query_backdoor_accepted
                else:
                    query_backdoor_accepted = True
                    query_accepted = (
                        query_gain_accepted
                        and query_clean_accepted
                    )
            elif self.meta_config.meta_objective == "query_gated_reptile":
                query_backdoor_accepted = _query_backdoor_constraints_accept(
                    base_backdoor=query_base_backdoor_acc,
                    adapted_backdoor=query_adapted_backdoor_acc,
                    backdoor_ceiling=self.meta_config.query_backdoor_ceiling,
                    backdoor_increase_tolerance=self.meta_config.query_backdoor_increase_tolerance,
                    backdoor_improvement_margin=self.meta_config.query_backdoor_improvement_margin,
                )
                query_accepted = (
                    query_gain_accepted
                    and query_clean_accepted
                    and query_backdoor_accepted
                )
            else:
                if str(attack_type.objective) == "targeted":
                    query_backdoor_accepted = _query_backdoor_reduction_accept(
                        base_backdoor=query_base_backdoor_acc,
                        adapted_backdoor=query_adapted_backdoor_acc,
                        reduction_margin=self.meta_config.query_targeted_asr_reduction_margin,
                        min_base_backdoor=self.meta_config.query_targeted_min_base_backdoor,
                    )
                else:
                    query_backdoor_accepted = True
                query_accepted = query_gain_accepted and query_clean_accepted and query_backdoor_accepted
            diagnostics.update(
                {
                    "query_diagnostics_only": float(diagnostics_only),
                    "query_base_reward": query_base_reward,
                    "query_adapted_reward": query_adapted_reward,
                    "query_gain": query_gain,
                    "query_base_clean_acc": query_base_clean_acc,
                    "query_adapted_clean_acc": query_adapted_clean_acc,
                    "query_clean_drop": query_clean_drop,
                    "query_base_backdoor_acc": query_base_backdoor_acc,
                    "query_adapted_backdoor_acc": query_adapted_backdoor_acc,
                    "query_backdoor_reduction": query_base_backdoor_acc - query_adapted_backdoor_acc,
                    "query_gain_accepted": float(query_gain_accepted),
                    "query_clean_accepted": float(query_clean_accepted),
                    "query_backdoor_accepted": float(query_backdoor_accepted),
                    "query_accepted": float(query_accepted),
                }
            )

        return TaskResult(
            attack_type=attack_type,
            adapted_params=adapted_p,
            mean_defender_reward=reward_sum_D / max(1, transition_count),
            mean_attacker_reward=reward_sum_A / max(1, transition_count),
            defender_reward_sum=reward_sum_D,
            trajectories_collected=trajectory_count,
            transitions_collected=transition_count,
            defender_losses=_aggregate_losses(all_def_losses),
            attacker_br_losses=attacker_br_losses,
            inner_delta_norm=inner_delta_norm,
            query_base_reward=query_base_reward,
            query_adapted_reward=query_adapted_reward,
            query_gain=query_gain,
            query_base_clean_acc=query_base_clean_acc,
            query_adapted_clean_acc=query_adapted_clean_acc,
            query_clean_drop=query_clean_drop,
            query_base_backdoor_acc=query_base_backdoor_acc,
            query_adapted_backdoor_acc=query_adapted_backdoor_acc,
            query_gain_accepted=query_gain_accepted,
            query_clean_accepted=query_clean_accepted,
            query_backdoor_accepted=query_backdoor_accepted,
            query_accepted=query_accepted,
            diagnostics=diagnostics,
        )

    def _build_env(self, attack_type: AttackType, horizon: int, seed: int | None = None) -> BSMGEnv:
        attack_strategy = self._build_attack_strategy(attack_type)
        coordinator = _coordinator_from_factory(
            self.coordinator_factory,
            attack_type=attack_type,
            horizon=max(1, int(horizon)),
            seed=seed,
        )
        post_training_evaluator = _post_training_evaluator_for_config(self.meta_config, coordinator)
        return BSMGEnv(
            coordinator=coordinator,
            attack_type=attack_type,
            attack_strategy=attack_strategy,
            defense_strategy=PaperDefenseStrategy(),
            config=BSMGConfig(
                horizon=max(1, int(horizon)),
                eval_every=self.meta_config.eval_every,
                history_len=self.meta_config.history_len,
                lambda_bd=self.meta_config.lambda_bd,
                reward_mode=self.meta_config.reward_mode,
                third_action=self.meta_config.defender_third_action,
                eps_min=self.meta_config.eps_min,
                eps_max=self.meta_config.eps_max,
                eps_log_scale=self.meta_config.eps_log_scale,
                server_lr_min=self.meta_config.server_lr_min,
                server_lr_max=self.meta_config.server_lr_max,
                server_lr_penalty_weight=self.meta_config.server_lr_penalty_weight,
                attack_context_names=self.meta_config.attack_context_names,
            ),
            evaluator=None
            if post_training_evaluator is not None
            else getattr(coordinator, "evaluate_weights", None),
            post_training_evaluator=post_training_evaluator,
        )

    def _query_evaluate(
        self,
        attack_type: AttackType,
        defender: TD3Agent,
        horizon: int,
        seed: int,
    ) -> dict[str, float]:
        env = self._build_env(attack_type, horizon=horizon, seed=seed)
        collector = TrajectoryCollector(
            env=env,
            defender=defender,
            attacker=self._rollout_attacker_policy(attack_type),
            defender_buffer=ReplayBuffer(
                max(1, int(horizon)),
                self.obs_dim,
                self.act_dim,
            ),
            attacker_buffer=None,
            exploration_noise=0.0,
            store_attacker=False,
        )
        traj = collector.collect(max(1, int(horizon)), seed=seed)
        return _query_rollout_metrics(traj)

    def _build_attack_strategy(self, attack_type: AttackType) -> AttackStrategy | None:
        if attack_type.name == "clean":
            return None
        if self.meta_config.native_sandbox_attacks and attack_type.name in {"bfl", "dba", "rl_backdoor", "mixed_backdoor"}:
            return NativeSandboxAttackMarker(attack_type)
        if attack_type.adaptive:
            return AdaptiveAttackStrategy(attack_type, self.attacker_agents[attack_type.name])
        return build_fixed_attack(attack_type)

    def _rollout_attacker_policy(self, attack_type: AttackType):
        if attack_type.adaptive:
            return self.attacker_agents[attack_type.name]
        return ConstantActionPolicy(act_dim=self.attacker_act_dim)

    def _warmup_steps(self) -> int:
        if self.meta_config.warmup_steps is not None:
            return max(0, self.meta_config.warmup_steps)
        return max(0, min(self.td3_config.warmup_steps, self.meta_config.H))


# ── Trajectory diagnostic helpers ────────────────────────────────────────────

def _action_diagnostics(traj: Trajectory) -> Dict[str, float]:
    """Per-dimension mean and std of defender actions over the trajectory."""
    if not traj.transitions:
        return {}
    actions = np.stack([tr.defender_action for tr in traj.transitions], axis=0)
    raw_std  = np.std(actions,  axis=0)
    decisions = [
        tr.info.get("defense_decision")
        for tr in traj.transitions
        if isinstance(tr.info.get("defense_decision"), DefenseDecision)
    ]
    if decisions:
        diagnostics = {
            "defender_alpha": float(np.mean([d.norm_bound_alpha for d in decisions])),
            "defender_beta": float(np.mean([d.trimmed_mean_beta for d in decisions])),
        }
        server_lrs = [d.server_lr for d in decisions if d.server_lr is not None]
        if server_lrs:
            diagnostics["defender_server_lr"] = float(np.mean(server_lrs))
        post_params = [
            d.neuroclip_epsilon
            if d.neuroclip_epsilon is not None
            else d.prun_mask_rate
            for d in decisions
            if d.neuroclip_epsilon is not None or d.prun_mask_rate is not None
        ]
        if post_params:
            diagnostics["defender_post_param"] = float(np.mean(post_params))
    else:
        raw_mean = np.mean(actions, axis=0)
        clipped  = np.clip(raw_mean, -1.0, 1.0)
        diagnostics = {
            "defender_alpha":       float((clipped[0] + 1.0) / 2.0 * 5.0),
            "defender_beta":        float((clipped[1] + 1.0) / 2.0 * 0.45),
            "defender_post_param":  float((clipped[2] + 1.0) / 2.0 * 10.0),
        }
    diagnostics.update({
        # Raw action std per dimension (exploration diversity)
        **{
            f"defender_action_std_{idx}": float(raw_std[idx])
            for idx in range(raw_std.shape[0])
        },
    })
    server_lr_penalties = [
        float(tr.info["server_lr_penalty"])
        for tr in traj.transitions
        if "server_lr_penalty" in tr.info
    ]
    if server_lr_penalties:
        diagnostics["server_lr_penalty"] = float(np.mean(server_lr_penalties))
    return diagnostics


def _attacker_action_diagnostics(traj: Trajectory) -> Dict[str, float]:
    """Per-trajectory decoded adaptive-attacker action statistics."""
    if not traj.transitions:
        return {}
    actions = np.stack([tr.attacker_action for tr in traj.transitions], axis=0)
    raw_mean = np.mean(actions, axis=0)
    raw_std = np.std(actions, axis=0)
    decision = AttackDecision.from_raw(raw_mean)
    return {
        "attacker_gamma": float(decision.gamma_scale),
        "attacker_local_steps": float(decision.local_steps),
        "attacker_lambda_stealth": float(decision.lambda_stealth),
        "attacker_action_std_0": float(raw_std[0]) if raw_std.shape[0] > 0 else 0.0,
        "attacker_action_std_1": float(raw_std[1]) if raw_std.shape[0] > 1 else 0.0,
        "attacker_action_std_2": float(raw_std[2]) if raw_std.shape[0] > 2 else 0.0,
    }


def _env_diagnostics(traj: Trajectory) -> Dict[str, float]:
    """Mean clean_acc, backdoor_acc, and attacker reward from trajectory info."""
    if not traj.transitions:
        return {}
    clean_accs = [tr.info.get("clean_acc", float("nan"))    for tr in traj.transitions]
    bd_accs    = [tr.info.get("backdoor_acc", float("nan")) for tr in traj.transitions]
    return {
        "clean_acc":    float(np.nanmean(clean_accs)),
        "backdoor_acc": float(np.nanmean(bd_accs)),
    }


def _query_rollout_metrics(traj: Trajectory) -> dict[str, float]:
    if not traj.transitions:
        return {
            "reward": float("nan"),
            "clean_acc": float("nan"),
            "backdoor_acc": float("nan"),
        }
    final = traj.transitions[-1]
    return {
        "reward": float(np.mean([tr.defender_reward for tr in traj.transitions])),
        "clean_acc": _finite_info(final, "clean_acc"),
        "backdoor_acc": _finite_info(final, "backdoor_acc"),
    }


def _finite_info(transition: Transition, key: str) -> float:
    value = transition.info.get(key, float("nan"))
    return float(value)


def _post_training_evaluator_for_config(meta_config: MetaSGConfig, coordinator: FLCoordinator):
    if str(getattr(meta_config, "post_defense_mode", "weight_copy")) != "model_aware_neuroclip":
        return None
    runner = getattr(coordinator, "runner", None)
    evaluate_model = getattr(coordinator, "evaluate_model", None)
    if runner is None or evaluate_model is None:
        raise ValueError("model_aware_neuroclip requires coordinator.runner.model and evaluate_model().")

    def evaluator(weights, decision: DefenseDecision) -> dict[str, float]:
        epsilon = decision.neuroclip_epsilon
        if epsilon is None:
            evaluate_weights = getattr(coordinator, "evaluate_weights", None)
            if evaluate_weights is None:
                raise ValueError("No NeuroClip epsilon was provided and coordinator cannot evaluate weights.")
            return evaluate_weights(weights)
        model = getattr(runner, "model", None)
        if model is None:
            raise ValueError("model_aware_neuroclip requires coordinator.runner.model at evaluation time.")
        defended_model = apply_post_defense(model, "neuroclip", float(epsilon))
        return evaluate_model(defended_model, weights)

    return evaluator


def _clean_drop(base_clean: float, adapted_clean: float) -> float:
    if math.isnan(base_clean) or math.isnan(adapted_clean):
        return float("nan")
    return float(base_clean - adapted_clean)


def _coordinator_from_factory(
    factory: Callable[..., FLCoordinator],
    *,
    attack_type: AttackType,
    horizon: int,
    seed: int | None,
) -> FLCoordinator:
    """Call attack-aware factories while preserving older zero-arg factories."""
    try:
        params = signature(factory).parameters
    except (TypeError, ValueError):
        return factory()

    accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
    kwargs = {}
    for key, value in {
        "attack_type": attack_type,
        "horizon": int(horizon),
        "seed": seed,
    }.items():
        if accepts_kwargs or key in params:
            kwargs[key] = value
    return factory(**kwargs)


def _query_clean_constraints_accept(
    *,
    base_clean: float,
    adapted_clean: float,
    clean_floor: float | None,
    clean_drop_tolerance: float | None,
) -> bool:
    if clean_floor is not None:
        if math.isnan(adapted_clean) or adapted_clean < float(clean_floor):
            return False
    if clean_drop_tolerance is not None:
        drop = _clean_drop(base_clean, adapted_clean)
        if math.isnan(drop) or drop > float(clean_drop_tolerance):
            return False
    return True


def _query_backdoor_constraints_accept(
    *,
    base_backdoor: float,
    adapted_backdoor: float,
    backdoor_ceiling: float | None,
    backdoor_increase_tolerance: float | None,
    backdoor_improvement_margin: float | None = None,
) -> bool:
    eps = 1e-12
    if (
        backdoor_ceiling is None
        and backdoor_increase_tolerance is None
        and backdoor_improvement_margin is None
    ):
        return True
    if not math.isfinite(adapted_backdoor):
        return False
    if backdoor_ceiling is not None and adapted_backdoor > float(backdoor_ceiling) + eps:
        if (
            backdoor_improvement_margin is None
            or not math.isfinite(base_backdoor)
            or base_backdoor <= float(backdoor_ceiling) + eps
            or base_backdoor - adapted_backdoor < float(backdoor_improvement_margin) - eps
        ):
            return False
    if backdoor_increase_tolerance is not None:
        if not math.isfinite(base_backdoor):
            return False
        if adapted_backdoor - base_backdoor > float(backdoor_increase_tolerance) + eps:
            return False
    return True


def _query_backdoor_reduction_accept(
    *,
    base_backdoor: float,
    adapted_backdoor: float,
    reduction_margin: float | None,
    min_base_backdoor: float | None,
) -> bool:
    eps = 1e-12
    if not math.isfinite(base_backdoor) or not math.isfinite(adapted_backdoor):
        return False
    if min_base_backdoor is not None and base_backdoor < float(min_base_backdoor) - eps:
        return False
    margin = 0.0 if reduction_margin is None else float(reduction_margin)
    return bool(base_backdoor - adapted_backdoor >= margin - eps)


def _aggregate_losses(loss_dicts: List[Dict]) -> Dict[str, float]:
    """Mean losses over all gradient steps; actor_loss excludes NaN (delayed update)."""
    if not loss_dicts:
        return {}
    critic_vals = [d["critic_loss"] for d in loss_dicts if "critic_loss" in d]
    q_vals      = [d["q_mean"]      for d in loss_dicts if "q_mean"      in d]
    actor_vals  = [d["actor_loss"]  for d in loss_dicts
                   if "actor_loss" in d and not math.isnan(d["actor_loss"])]
    result: Dict[str, float] = {}
    if critic_vals:
        result["critic_loss"] = float(np.mean(critic_vals))
    if actor_vals:
        result["actor_loss"] = float(np.mean(actor_vals))
    if q_vals:
        result["q_mean"] = float(np.mean(q_vals))
    return result


def _merge_weighted(
    left: Dict[str, float],
    right: Dict[str, float],
    left_count: int,
    right_count: int,
) -> Dict[str, float]:
    if not left:
        return right
    if not right:
        return left
    total = max(1, left_count + right_count)
    return {
        key: (left.get(key, 0.0) * left_count + right.get(key, 0.0) * right_count) / total
        for key in set(left) | set(right)
    }
