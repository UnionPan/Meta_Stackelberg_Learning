"""Diagnose whether Meta-SG can learn attack-conditional defenses.

This script answers two questions:

1. Do different attack types prefer different fixed defense actions?
2. Do observations contain enough signal to separate attack types?

Run:
    .venv/bin/python -u meta_sg/scripts/diagnose_defense_learning.py \
      --num-clients 100 --num-attackers 20 --subsample-rate 0.1 \
      --H 4 --seeds 201
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_sandbox.federation.runner import SandboxConfig
from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.evaluation import PolicyEvaluator
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter
from meta_sg.strategies.attacks.fixed import build_fixed_attack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN, DefenseDecision


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_key: str
    patch: dict


@dataclass(frozen=True)
class ActionSpec:
    name: str
    raw: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--client-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--seeds", default="201")
    parser.add_argument("--history-len", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    base_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device="cpu",
        seed=42,
        num_clients=args.num_clients,
        num_attackers=args.num_attackers,
        subsample_rate=args.subsample_rate,
        local_epochs=1,
        lr=0.05,
        batch_size=32,
        eval_batch_size=256,
        fltrust_root_size=0,
        max_client_samples_per_client=args.client_samples,
        max_eval_samples=args.eval_samples,
        parallel_clients=1,
    )

    print("CONFIG", {
        "H": args.H,
        "num_clients": args.num_clients,
        "num_attackers": args.num_attackers,
        "subsample_rate": args.subsample_rate,
        "client_samples": args.client_samples,
        "eval_samples": args.eval_samples,
        "seeds": seeds,
        "history_len": args.history_len,
    })
    run_action_sweep(base_config, args.H, seeds, history_len=args.history_len)
    run_observation_separability(base_config, args.H, seeds, history_len=args.history_len)


def run_action_sweep(base_config: SandboxConfig, horizon: int, seeds: Iterable[int], history_len: int) -> None:
    scenarios = [
        Scenario("clean", "ipm", {"num_attackers": 0, "ipm_scaling": 0.0}),
        Scenario("ipm", "ipm", {"ipm_scaling": 2.0}),
        Scenario("lmp", "lmp", {"lmp_scale": 2.0}),
        Scenario("bfl", "bfl", {"bfl_poison_frac": 1.0}),
        Scenario("dba", "dba", {"dba_poison_frac": 0.5, "dba_num_sub_triggers": 4}),
    ]
    actions = [
        ActionSpec("no_robust_agg", (1.0, -1.0, 0.0)),
        ActionSpec("low_trim", (0.0, -1.0, 0.0)),
        ActionSpec("mid_action", (0.0, 0.0, 0.0)),
        ActionSpec("learned_like", (-0.14, -0.05, -0.08)),
        ActionSpec("strong_trim", (0.0, 1.0, 0.0)),
        ActionSpec("strong_clip_trim", (-0.7, 1.0, 0.0)),
    ]

    print("\nACTION_SWEEP scenario action reward clean backdoor attack_damage alpha beta post")
    best_by_scenario: dict[str, tuple[str, float]] = {}
    for scenario in scenarios:
        config = _patched_config(base_config, scenario.patch)
        coordinator_factory = lambda cfg=config: FLSandboxCoordinatorAdapter(cfg)
        obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights(), history_len=history_len)
        evaluator = PolicyEvaluator(
            coordinator_factory=coordinator_factory,
            horizon=horizon,
            obs_dim=obs_dim,
            eval_every=1,
            history_len=history_len,
        )
        attack_type = ATTACK_DOMAIN[scenario.attack_key]
        for action in actions:
            summary = evaluator.evaluate(
                action.name,
                ConstantActionPolicy(np.asarray(action.raw, dtype=np.float32)),
                [attack_type],
                seeds=tuple(seeds),
            )
            damage = _attack_damage(scenario.name, attack_type.objective, summary.mean_final_clean_acc, summary.mean_final_backdoor_acc)
            decision = DefenseDecision.from_raw(np.asarray(action.raw, dtype=np.float32))
            print(
                scenario.name,
                action.name,
                round(summary.mean_reward, 4),
                round(summary.mean_final_clean_acc, 4),
                round(summary.mean_final_backdoor_acc, 4),
                round(damage, 4) if np.isfinite(damage) else "nan",
                round(decision.norm_bound_alpha, 4),
                round(decision.trimmed_mean_beta, 4),
                round(decision.neuroclip_epsilon or 0.0, 4),
            )
            current = best_by_scenario.get(scenario.name)
            if current is None or summary.mean_reward > current[1]:
                best_by_scenario[scenario.name] = (action.name, summary.mean_reward)

    print("\nACTION_BEST_BY_REWARD")
    for scenario, (action, reward) in best_by_scenario.items():
        print(scenario, action, round(reward, 4))


def run_observation_separability(base_config: SandboxConfig, horizon: int, seeds: Iterable[int], history_len: int) -> None:
    attack_keys = ("ipm", "lmp", "bfl", "dba")
    X: list[np.ndarray] = []
    y: list[int] = []
    labels: list[str] = []
    for class_id, attack_key in enumerate(attack_keys):
        labels.append(attack_key)
        for seed in seeds:
            traj = _collect_mid_action_trajectory(base_config, attack_key, horizon, seed, history_len)
            for transition in traj.transitions:
                X.append(transition.next_state.astype(np.float32))
                y.append(class_id)

    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)
    acc, confusion = _nearest_centroid_cv(X_arr, y_arr, len(labels))
    centroid_dist = _centroid_distance_table(X_arr, y_arr, labels)

    print("\nOBS_SEPARABILITY nearest_centroid_leave_one_out_acc", round(acc, 4))
    print("OBS_CONFUSION rows=true cols=pred labels=" + ",".join(labels))
    for label, row in zip(labels, confusion):
        print(label, " ".join(str(int(v)) for v in row))
    print("OBS_CENTROID_DISTANCE")
    for row in centroid_dist:
        print(" ".join(row))


def _collect_mid_action_trajectory(
    base_config: SandboxConfig,
    attack_key: str,
    horizon: int,
    seed: int,
    history_len: int,
):
    coordinator = FLSandboxCoordinatorAdapter(base_config)
    attack_type = ATTACK_DOMAIN[attack_key]
    env = BSMGEnv(
        coordinator=coordinator,
        attack_type=attack_type,
        attack_strategy=build_fixed_attack(attack_type),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=horizon, eval_every=1, history_len=history_len),
    )
    obs_dim = env.obs_dim
    collector = TrajectoryCollector(
        env=env,
        defender=ConstantActionPolicy([0.0, 0.0, 0.0]),
        attacker=ConstantActionPolicy(act_dim=3),
        defender_buffer=ReplayBuffer(max(horizon, 1), obs_dim, 3),
        attacker_buffer=ReplayBuffer(max(horizon, 1), obs_dim, 3),
        exploration_noise=0.0,
        store_attacker=False,
    )
    return collector.collect(horizon, seed=seed)


def _nearest_centroid_cv(X: np.ndarray, y: np.ndarray, num_classes: int) -> tuple[float, np.ndarray]:
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        centroids = []
        for cls in range(num_classes):
            cls_X = X[mask & (y == cls)]
            centroids.append(np.mean(cls_X, axis=0))
        centroids_arr = np.stack(centroids, axis=0)
        distances = np.linalg.norm(centroids_arr - X[i], axis=1)
        preds.append(int(np.argmin(distances)))
    pred_arr = np.asarray(preds, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y, pred_arr):
        confusion[int(true), int(pred)] += 1
    return float(np.mean(pred_arr == y)), confusion


def _centroid_distance_table(X: np.ndarray, y: np.ndarray, labels: list[str]) -> list[list[str]]:
    centroids = np.stack([np.mean(X[y == cls], axis=0) for cls in range(len(labels))], axis=0)
    rows = [["."] + labels]
    for i, label in enumerate(labels):
        row = [label]
        for j in range(len(labels)):
            row.append(str(round(float(np.linalg.norm(centroids[i] - centroids[j])), 4)))
        rows.append(row)
    return rows


def _patched_config(base: SandboxConfig, patch: dict) -> SandboxConfig:
    values = vars(base).copy()
    values.update(patch)
    return SandboxConfig(**values)


def _attack_damage(scenario: str, objective: str, clean_acc: float, backdoor_acc: float) -> float:
    if scenario == "clean":
        return float("nan")
    if objective == "targeted":
        return float(backdoor_acc)
    return float(1.0 - clean_acc)


if __name__ == "__main__":
    main()
