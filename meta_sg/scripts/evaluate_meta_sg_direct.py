"""Direct final-accuracy evaluation for a trained Meta-SG defender.

This script fixes a learned defender checkpoint and evaluates it in the same
H-step FL setting used by the fixed-defense baseline table. It intentionally
reports final clean accuracy, not the training-side rollout reward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackType


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_type: AttackType
    attack_name: str
    patch: dict
    seed: int


class SandboxAttackMarker:
    """Name-only marker that lets FLSandboxCoordinatorAdapter build native attacks."""

    def __init__(self, attack_type: AttackType) -> None:
        self.attack_type = attack_type
        self.name = attack_type.name


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to defender_meta.pt or checkpoint directory.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=4)
    parser.add_argument("--subsample-rate", type=float, default=0.5)
    parser.add_argument("--client-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=502)
    parser.add_argument("--rl-seed", type=int, default=506)
    parser.add_argument(
        "--rl-policy-checkpoint",
        default=(
            "runs/meta_sg_goal/fl_medium_h20_rl_clipped_median_eval/"
            "outputs/mnist_rl_clipped_median_iid_20r/checkpoints/rl_policy_latest.pt"
        ),
    )
    parser.add_argument(
        "--rl-distribution-dir",
        default="fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1_init_pre_label",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    probe = FLSandboxCoordinatorAdapter(_sandbox_config(args, "clean", seed=args.seed, patch={"num_attackers": 0}))
    obs_dim = BSMGEnv(
        coordinator=probe,
        attack_type=_attack_type("clean"),
        attack_strategy=None,
        defense_strategy=PaperDefenseStrategy(),
        config=_bsmg_config(args),
        evaluator=getattr(probe, "evaluate_weights", None),
    ).reset(seed=args.seed).shape[0]

    defender = TD3Agent(
        obs_dim,
        3,
        TD3Config(hidden_dim=args.hidden_dim, batch_size=16, buffer_capacity=4096, warmup_steps=0),
        device=device,
    )
    defender.load(_checkpoint_path(args.checkpoint))

    scenarios = [
        Scenario("clean", _attack_type("clean"), "clean", {"num_attackers": 0}, int(args.seed)),
        Scenario("ipm", _attack_type("ipm"), "ipm", {"ipm_scaling": 2.0}, int(args.seed)),
        Scenario("lmp", _attack_type("lmp"), "lmp", {"lmp_scale": 2.0}, int(args.seed)),
        Scenario("rl", _attack_type("rl"), "rl", _rl_patch(args), int(args.rl_seed)),
    ]

    records = []
    for scenario in scenarios:
        record = _evaluate_scenario(args, defender, scenario)
        records.append(record)
        print(
            scenario.name,
            "final_clean_acc=",
            round(record["final_clean_acc"], 4),
            "mean_reward=",
            round(record["mean_defender_reward"], 4),
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_scenario(args, defender: TD3Agent, scenario: Scenario) -> dict:
    coordinator = FLSandboxCoordinatorAdapter(
        _sandbox_config(args, scenario.attack_name, seed=scenario.seed, patch=scenario.patch)
    )
    attack_strategy = None if scenario.attack_name == "clean" else SandboxAttackMarker(scenario.attack_type)
    env = BSMGEnv(
        coordinator=coordinator,
        attack_type=scenario.attack_type,
        attack_strategy=attack_strategy,
        defense_strategy=PaperDefenseStrategy(),
        config=_bsmg_config(args),
        evaluator=getattr(coordinator, "evaluate_weights", None),
    )
    obs = env.reset(seed=scenario.seed)
    rewards_d: list[float] = []
    rewards_a: list[float] = []
    last_info = {}
    for _ in range(int(args.H)):
        defender_action = defender.get_action(obs, noise=0.0)
        attacker_action = np.zeros(3, dtype=np.float32)
        obs, r_d, r_a, done, info = env.step(defender_action, attacker_action)
        rewards_d.append(float(r_d))
        rewards_a.append(float(r_a))
        last_info = dict(info)
        if done:
            break
    return {
        "scenario": scenario.name,
        "attack_type": scenario.attack_name,
        "seed": int(scenario.seed),
        "horizon": int(args.H),
        "final_clean_acc": float(last_info.get("clean_acc", float("nan"))),
        "final_backdoor_acc": float(last_info.get("backdoor_acc", float("nan"))),
        "final_defender_reward": float(rewards_d[-1]) if rewards_d else float("nan"),
        "mean_defender_reward": float(np.mean(rewards_d)) if rewards_d else float("nan"),
        "mean_attacker_reward": float(np.mean(rewards_a)) if rewards_a else float("nan"),
        "final_defender_alpha": float(last_info.get("defense_decision").norm_bound_alpha)
        if last_info.get("defense_decision") is not None
        else float("nan"),
        "final_defender_beta": float(last_info.get("defense_decision").trimmed_mean_beta)
        if last_info.get("defense_decision") is not None
        else float("nan"),
    }


def _sandbox_config(args, attack_name: str, *, seed: int, patch: dict) -> object:
    values = {
        "dataset": args.dataset,
        "attack_type": attack_name,
        "defense_type": "paper_norm_trimmed_mean",
        "rounds": int(args.H),
        "num_clients": int(args.num_clients),
        "num_attackers": int(args.num_attackers),
        "subsample_rate": float(args.subsample_rate),
        "seed": int(seed),
        "device": str(_resolve_device(args.device)),
        "parallel_clients": 1,
        "num_workers": 0,
        "local_epochs": 1,
        "lr": 0.05,
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "max_client_samples_per_client": int(args.client_samples),
        "max_eval_samples": int(args.eval_samples),
        "fltrust_root_size": 0,
        "krum_attackers": int(args.num_attackers),
        "trimmed_mean_ratio": 0.2,
        "clipped_median_norm": 2.0,
    }
    values.update(patch)
    return SandboxConfig(**values)


def _bsmg_config(args) -> BSMGConfig:
    return BSMGConfig(
        horizon=int(args.H),
        eval_every=1,
        history_len=0,
        lambda_bd=0.0,
        reward_mode="accuracy",
        third_action="neuroclip",
    )


def _rl_patch(args) -> dict:
    return {
        "rl_distribution_dir": str(args.rl_distribution_dir),
        "rl_attack_start_round": 6,
        "rl_policy_train_end_round": int(args.H),
        "rl_distribution_steps": 5,
        "rl_inversion_steps": 20,
        "rl_reconstruction_batch_size": 8,
        "rl_simulator_horizon": 5,
        "rl_policy_train_episodes_per_round": 1,
        "rl_policy_warmup_steps": 0,
        "rl_policy_warmup_random_steps": 0,
        "rl_policy_checkpoint_path": str(args.rl_policy_checkpoint),
    }


def _attack_type(name: str) -> AttackType:
    return AttackType(name=name, objective="untargeted", adaptive=(name == "rl"))


def _checkpoint_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "defender_meta.pt"
    return str(candidate)


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
