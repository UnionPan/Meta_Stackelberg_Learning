"""Policy-vs-attack evaluation matrix for the real fl_sandbox backend.

The goal is to separate three questions that a single reward curve hides:
  1. Does a defender preserve clean utility when the attacker is weak/absent?
  2. Does it improve robustness when the attacker is strong?
  3. Does adaptation help against adaptive attacker policies?

Run:
    /Users/antik/anaconda3/envs/gym_env/bin/python -u \
      meta_sg/scripts/evaluate_fl_sandbox_matrix.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from torch.utils.tensorboard import SummaryWriter

from fl_sandbox.federation.runner import SandboxConfig
from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.config import TD3Config
from meta_sg.learning.evaluation import PolicyEvalSummary, PolicyEvaluator
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter
from meta_sg.strategies.types import ATTACK_DOMAIN, AttackType


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_type: AttackType
    config_patch: dict
    adaptive: bool = False


@dataclass(frozen=True)
class DefenderSpec:
    name: str
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--client-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seeds", default="201,202,203")
    parser.add_argument("--include-rl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.abspath(os.path.join("runs", "meta_sg_fl_sandbox_matrix", run_name))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    base_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device="cpu",
        seed=42,
        num_clients=4,
        num_attackers=1,
        subsample_rate=1.0,
        local_epochs=1,
        lr=0.05,
        batch_size=32,
        eval_batch_size=256,
        fltrust_root_size=0,
        max_client_samples_per_client=args.client_samples,
        max_eval_samples=args.eval_samples,
        parallel_clients=1,
        # Keep optional RL attacker cheap when --include-rl is used.
        rl_attack_start_round=1,
        rl_policy_train_end_round=2,
        rl_distribution_steps=1,
        rl_inversion_steps=1,
        rl_reconstruction_batch_size=2,
        rl_policy_train_episodes_per_round=1,
        rl_simulator_horizon=2,
    )

    scenarios = [
        Scenario("clean_no_attacker", ATTACK_DOMAIN["ipm"], {"num_attackers": 0, "ipm_scaling": 0.0}),
        Scenario("weak_ipm", ATTACK_DOMAIN["ipm"], {"num_attackers": 1, "ipm_scaling": 0.5}),
        Scenario("paper_ipm", ATTACK_DOMAIN["ipm"], {"num_attackers": 1, "ipm_scaling": 2.0}),
        Scenario("strong_ipm", ATTACK_DOMAIN["ipm"], {"num_attackers": 1, "ipm_scaling": 4.0}),
        Scenario("bfl_backdoor", ATTACK_DOMAIN["bfl"], {"num_attackers": 1, "bfl_poison_frac": 1.0}),
    ]
    if args.include_rl:
        scenarios.append(Scenario("rl_adaptive_smoke", ATTACK_DOMAIN["rl"], {"num_attackers": 1}, adaptive=True))

    defenders = [
        DefenderSpec("mid_action", lambda: ConstantActionPolicy([0.0, 0.0, 0.0])),
        DefenderSpec("low_trim", lambda: ConstantActionPolicy([0.0, -1.0, 0.0])),
        DefenderSpec("strong_trim", lambda: ConstantActionPolicy([0.0, 1.0, 0.0])),
        DefenderSpec("strong_clip_trim", lambda: ConstantActionPolicy([-0.7, 1.0, 0.0])),
    ]

    print("LOG_DIR", log_dir)
    print("SCENARIO DEFENDER reward clean attack_damage clean_delta_vs_mid utility_cost_vs_mid gain_vs_mid")

    for scenario in scenarios:
        config = _patched_config(base_config, scenario.config_patch)
        coordinator_factory = lambda cfg=config: FLSandboxCoordinatorAdapter(cfg)
        obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights())
        evaluator = PolicyEvaluator(
            coordinator_factory=coordinator_factory,
            horizon=args.H,
            obs_dim=obs_dim,
            eval_every=1,
        )
        attacker_agents = _attacker_agents(obs_dim) if scenario.adaptive else None

        summaries: dict[str, PolicyEvalSummary] = {}
        for defender in defenders:
            summaries[defender.name] = evaluator.evaluate(
                defender.name,
                defender.factory(),
                [scenario.attack_type],
                attacker_agents=attacker_agents,
                seeds=seeds,
            )

        baseline = summaries["mid_action"]
        for defender_name, summary in summaries.items():
            clean_delta = summary.mean_final_clean_acc - baseline.mean_final_clean_acc
            utility_cost = max(0.0, -clean_delta)
            gain = summary.mean_reward - baseline.mean_reward
            attack_damage = _attack_damage(summary, scenario)
            prefix = f"matrix/{scenario.name}/{defender_name}"
            writer.add_scalar(f"{prefix}/mean_reward", summary.mean_reward, 0)
            writer.add_scalar(f"{prefix}/std_reward", summary.std_reward, 0)
            writer.add_scalar(f"{prefix}/worst_reward", summary.worst_reward, 0)
            writer.add_scalar(f"{prefix}/clean_acc", summary.mean_final_clean_acc, 0)
            writer.add_scalar(f"{prefix}/attack_damage", attack_damage, 0)
            writer.add_scalar(f"{prefix}/attacker_reward", summary.mean_attacker_reward, 0)
            writer.add_scalar(f"{prefix}/clean_delta_vs_mid", clean_delta, 0)
            writer.add_scalar(f"{prefix}/utility_cost_vs_mid", utility_cost, 0)
            writer.add_scalar(f"{prefix}/gain_vs_mid", gain, 0)
            print(
                scenario.name,
                defender_name,
                round(summary.mean_reward, 4),
                round(summary.mean_final_clean_acc, 4),
                round(attack_damage, 4),
                round(clean_delta, 4),
                round(utility_cost, 4),
                round(gain, 4),
            )

    writer.flush()
    writer.close()


def _patched_config(base: SandboxConfig, patch: dict) -> SandboxConfig:
    values = vars(base).copy()
    values.update(patch)
    return SandboxConfig(**values)


def _attacker_agents(obs_dim: int) -> dict[str, TD3Agent]:
    cfg = TD3Config(hidden_dim=32, batch_size=4, buffer_capacity=256, warmup_steps=0)
    return {"rl": TD3Agent(obs_dim, 3, cfg)}


def _attack_damage(summary: PolicyEvalSummary, scenario: Scenario) -> float:
    if scenario.name == "clean_no_attacker":
        return float("nan")
    if scenario.attack_type.objective == "targeted":
        return summary.mean_final_backdoor_acc
    return 1.0 - summary.mean_final_clean_acc


if __name__ == "__main__":
    main()
