"""Evaluation utilities for convergence and policy effectiveness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.games.trajectory import Trajectory
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.attacks.fixed import build_fixed_attack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType


@dataclass(frozen=True)
class ConvergenceReport:
    converged: bool
    rolling_mean: float
    rolling_std: float
    slope: float
    window: int
    reason: str


@dataclass(frozen=True)
class PolicyEvalMetrics:
    attack_name: str
    seed: int
    mean_defender_reward: float
    total_defender_reward: float
    final_defender_reward: float
    mean_attacker_reward: float
    final_clean_acc: float
    final_backdoor_acc: float
    horizon: int


@dataclass(frozen=True)
class PolicyEvalSummary:
    name: str
    metrics: list[PolicyEvalMetrics] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean([m.mean_defender_reward for m in self.metrics])) if self.metrics else float("nan")

    @property
    def std_reward(self) -> float:
        return float(np.std([m.mean_defender_reward for m in self.metrics])) if self.metrics else float("nan")

    @property
    def worst_reward(self) -> float:
        return float(np.min([m.mean_defender_reward for m in self.metrics])) if self.metrics else float("nan")

    @property
    def mean_final_clean_acc(self) -> float:
        return float(np.nanmean([m.final_clean_acc for m in self.metrics])) if self.metrics else float("nan")

    @property
    def mean_final_backdoor_acc(self) -> float:
        return float(np.nanmean([m.final_backdoor_acc for m in self.metrics])) if self.metrics else float("nan")

    @property
    def mean_attacker_reward(self) -> float:
        return float(np.mean([m.mean_attacker_reward for m in self.metrics])) if self.metrics else float("nan")


def assess_convergence(
    values: Sequence[float],
    *,
    window: int = 10,
    min_points: int | None = None,
    std_tol: float = 0.02,
    slope_tol: float = 0.002,
) -> ConvergenceReport:
    """
    Heuristic convergence check for training/evaluation curves.

    A run is considered converged when the most recent window has both low
    variance and near-zero linear trend. This is a diagnostic, not a proof.
    """
    min_points = min_points or max(window * 2, window + 2)
    if len(values) < min_points:
        return ConvergenceReport(False, float("nan"), float("nan"), float("nan"), window, "not_enough_points")

    y = np.asarray(values[-window:], dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    slope = float(np.polyfit(x, y, deg=1)[0])
    rolling_mean = float(np.mean(y))
    rolling_std = float(np.std(y))
    converged = abs(slope) <= slope_tol and rolling_std <= std_tol
    reason = "stable" if converged else "unstable_or_trending"
    return ConvergenceReport(converged, rolling_mean, rolling_std, slope, window, reason)


class PolicyEvaluator:
    """Evaluate a defender policy against attack tasks and seeds."""

    def __init__(
        self,
        coordinator_factory: Callable[[], FLCoordinator],
        horizon: int,
        obs_dim: int,
        act_dim: int = 3,
        eval_every: int = 1,
    ) -> None:
        self.coordinator_factory = coordinator_factory
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eval_every = eval_every

    def evaluate(
        self,
        name: str,
        defender_policy,
        attack_types: Iterable[AttackType],
        *,
        attacker_agents: dict[str, TD3Agent] | None = None,
        seeds: Sequence[int] = (0,),
    ) -> PolicyEvalSummary:
        metrics: list[PolicyEvalMetrics] = []
        for attack_type in attack_types:
            for seed in seeds:
                env = self._build_env(attack_type, attacker_agents=attacker_agents)
                attacker_policy = self._attacker_policy(attack_type, attacker_agents=attacker_agents)
                def_buffer = ReplayBuffer(max(self.horizon, 1), self.obs_dim, self.act_dim)
                atk_buffer = ReplayBuffer(max(self.horizon, 1), self.obs_dim, self.act_dim)
                collector = TrajectoryCollector(
                    env=env,
                    defender=defender_policy,
                    attacker=attacker_policy,
                    defender_buffer=def_buffer,
                    attacker_buffer=atk_buffer,
                    exploration_noise=0.0,
                    store_attacker=attack_type.adaptive,
                )
                traj = collector.collect(self.horizon, seed=seed)
                metrics.append(self._metrics_from_trajectory(attack_type, seed, traj))
        return PolicyEvalSummary(name=name, metrics=metrics)

    def evaluate_after_adaptation(
        self,
        name: str,
        defender_agent: TD3Agent,
        attack_types: Iterable[AttackType],
        *,
        attacker_agents: dict[str, TD3Agent] | None = None,
        seeds: Sequence[int] = (0,),
        adaptation_horizon: int | None = None,
        adaptation_updates: int = 10,
        exploration_noise: float = 0.1,
    ) -> PolicyEvalSummary:
        """
        Evaluate a meta-policy after task-specific TD3 adaptation.

        This matches the meta-learning contract: θ_meta is an initialization,
        so validation should also measure θ_task after a small number of
        environment steps and replay-buffer updates on the target attack.
        """
        metrics: list[PolicyEvalMetrics] = []
        adapt_h = adaptation_horizon or self.horizon
        for attack_type in attack_types:
            for seed in seeds:
                adapted = defender_agent.clone()
                env = self._build_env(attack_type, attacker_agents=attacker_agents)
                attacker_policy = self._attacker_policy(attack_type, attacker_agents=attacker_agents)
                def_buffer = ReplayBuffer(max(adapt_h, 1), self.obs_dim, self.act_dim)
                atk_buffer = ReplayBuffer(max(adapt_h, 1), self.obs_dim, self.act_dim)
                collector = TrajectoryCollector(
                    env=env,
                    defender=adapted,
                    attacker=attacker_policy,
                    defender_buffer=def_buffer,
                    attacker_buffer=atk_buffer,
                    exploration_noise=exploration_noise,
                    store_attacker=attack_type.adaptive,
                )
                collector.collect(adapt_h, seed=seed)
                for _ in range(adaptation_updates):
                    adapted.update(def_buffer)

                eval_env = self._build_env(attack_type, attacker_agents=attacker_agents)
                eval_buffer = ReplayBuffer(max(self.horizon, 1), self.obs_dim, self.act_dim)
                eval_atk_buffer = ReplayBuffer(max(self.horizon, 1), self.obs_dim, self.act_dim)
                eval_collector = TrajectoryCollector(
                    env=eval_env,
                    defender=adapted,
                    attacker=attacker_policy,
                    defender_buffer=eval_buffer,
                    attacker_buffer=eval_atk_buffer,
                    exploration_noise=0.0,
                    store_attacker=attack_type.adaptive,
                )
                traj = eval_collector.collect(self.horizon, seed=seed + 50_000)
                metrics.append(self._metrics_from_trajectory(attack_type, seed, traj))
        return PolicyEvalSummary(name=name, metrics=metrics)

    def _build_env(
        self,
        attack_type: AttackType,
        *,
        attacker_agents: dict[str, TD3Agent] | None,
    ) -> BSMGEnv:
        if attack_type.adaptive:
            if attacker_agents is None or attack_type.name not in attacker_agents:
                raise ValueError(f"Missing adaptive attacker agent for {attack_type.name}")
            attack_strategy = AdaptiveAttackStrategy(attack_type, attacker_agents[attack_type.name])
        else:
            attack_strategy = build_fixed_attack(attack_type)
        return BSMGEnv(
            coordinator=self.coordinator_factory(),
            attack_type=attack_type,
            attack_strategy=attack_strategy,
            defense_strategy=PaperDefenseStrategy(),
            config=BSMGConfig(horizon=self.horizon, eval_every=self.eval_every),
        )

    def _attacker_policy(
        self,
        attack_type: AttackType,
        *,
        attacker_agents: dict[str, TD3Agent] | None,
    ):
        if attack_type.adaptive:
            if attacker_agents is None or attack_type.name not in attacker_agents:
                raise ValueError(f"Missing adaptive attacker agent for {attack_type.name}")
            return attacker_agents[attack_type.name]
        return ConstantActionPolicy(act_dim=self.act_dim)

    @staticmethod
    def _metrics_from_trajectory(
        attack_type: AttackType,
        seed: int,
        trajectory: Trajectory,
    ) -> PolicyEvalMetrics:
        rewards_d = [t.defender_reward for t in trajectory.transitions]
        rewards_a = [t.attacker_reward for t in trajectory.transitions]
        last = trajectory.transitions[-1]
        info = last.info
        return PolicyEvalMetrics(
            attack_name=attack_type.name,
            seed=seed,
            mean_defender_reward=float(np.mean(rewards_d)),
            total_defender_reward=float(np.sum(rewards_d)),
            final_defender_reward=float(rewards_d[-1]),
            mean_attacker_reward=float(np.mean(rewards_a)),
            final_clean_acc=float(info.get("clean_acc", float("nan"))),
            final_backdoor_acc=float(info.get("backdoor_acc", float("nan"))),
            horizon=trajectory.horizon,
        )
