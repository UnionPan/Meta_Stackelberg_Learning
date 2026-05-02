"""Per-attack task execution for Meta-SG pre-training."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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
from meta_sg.strategies.types import AttackType


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
    # Env-level diagnostics (action distribution, clean/backdoor acc)
    diagnostics: dict = field(default_factory=dict)


class AttackTaskRunner:
    """Runs one sampled attack task ξ from a common meta-policy start."""

    def __init__(
        self,
        coordinator_factory: Callable[[], FLCoordinator],
        td3_config: TD3Config,
        meta_config: MetaSGConfig,
        obs_dim: int,
        act_dim: int,
        attacker_agents: Dict[str, TD3Agent],
        attacker_buffers: Dict[str, ReplayBuffer],
        best_response: AttackerBestResponse,
    ) -> None:
        self.coordinator_factory = coordinator_factory
        self.td3_config = td3_config
        self.meta_config = meta_config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.attacker_agents = attacker_agents
        self.attacker_buffers = attacker_buffers
        self.best_response = best_response

    def run(self, attack_type: AttackType, meta_defender: TD3Agent, seed_base: int) -> TaskResult:
        local_defender = meta_defender.clone()
        local_def_buffer = ReplayBuffer(
            self.td3_config.buffer_capacity, self.obs_dim, self.act_dim
        )

        attack_strategy = self._build_attack_strategy(attack_type)
        env = BSMGEnv(
            coordinator=self.coordinator_factory(),
            attack_type=attack_type,
            attack_strategy=attack_strategy,
            defense_strategy=PaperDefenseStrategy(),
            config=BSMGConfig(
                horizon=self.meta_config.H,
                eval_every=self.meta_config.eval_every,
            ),
        )

        collector = TrajectoryCollector(
            env=env,
            defender=local_defender,
            attacker=self._rollout_attacker_policy(attack_type),
            defender_buffer=local_def_buffer,
            attacker_buffer=self.attacker_buffers[attack_type.name],
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

        # Phase 1: collect trajectory, then l TD3 gradient updates.
        traj = collector.collect(self.meta_config.H, seed=seed_base)
        action_diag = _action_diagnostics(traj)
        env_diag    = _env_diagnostics(traj)
        reward_sum_D += sum(t.defender_reward for t in traj.transitions)
        reward_sum_A += sum(t.attacker_reward for t in traj.transitions)
        transition_count += len(traj.transitions)
        trajectory_count += 1

        all_def_losses: List[Dict] = []
        for _ in range(self.meta_config.l):
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
            diagnostics={
                "buffer_size": len(local_def_buffer),
                "attacker_buffer_size": len(self.attacker_buffers[attack_type.name]),
                **action_diag,
                **env_diag,
            },
        )

    def _build_attack_strategy(self, attack_type: AttackType) -> AttackStrategy:
        if attack_type.adaptive:
            return AdaptiveAttackStrategy(attack_type, self.attacker_agents[attack_type.name])
        return build_fixed_attack(attack_type)

    def _rollout_attacker_policy(self, attack_type: AttackType):
        if attack_type.adaptive:
            return self.attacker_agents[attack_type.name]
        return ConstantActionPolicy(act_dim=self.act_dim)

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
    raw_mean = np.mean(actions, axis=0)
    raw_std  = np.std(actions,  axis=0)
    clipped  = np.clip(raw_mean, -1.0, 1.0)
    return {
        # Decoded physical parameters (for policy/defender/* group)
        "defender_alpha":       float((clipped[0] + 1.0) / 2.0 * 5.0),
        "defender_beta":        float((clipped[1] + 1.0) / 2.0 * 0.45),
        "defender_post_param":  float((clipped[2] + 1.0) / 2.0 * 10.0),
        # Raw action std per dimension (exploration diversity)
        "defender_action_std_0": float(raw_std[0]),
        "defender_action_std_1": float(raw_std[1]),
        "defender_action_std_2": float(raw_std[2]),
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
