"""Evaluate fixed FL aggregation defenses without RL actions.

This complements the Meta-SG matrix by testing ordinary FedAvg, Krum, Median,
and related robust aggregators through ``MinimalFLRunner`` directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_sandbox.attacks import create_attack
from fl_sandbox.federation.runner import MinimalFLRunner
from meta_sg.simulation.fl_sandbox_adapter import SandboxConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_type: str
    patch: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=4)
    parser.add_argument("--subsample-rate", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--client-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--defenses", default="fedavg,krum,median,trimmed_mean,clipped_median")
    parser.add_argument(
        "--scenarios",
        default="clean,paper_ipm,strong_ipm,lmp,bfl,dba",
        help="Comma-separated scenario names to run, in output order.",
    )
    parser.add_argument("--output-json", default="", help="Optional path for machine-readable records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=args.device,
        seed=args.seed,
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
        krum_attackers=args.num_attackers,
        trimmed_mean_ratio=0.2,
        clipped_median_norm=2.0,
    )
    scenarios = [
        Scenario("clean", "clean", {"num_attackers": 0}),
        Scenario("paper_ipm", "ipm", {"ipm_scaling": 2.0}),
        Scenario("strong_ipm", "ipm", {"ipm_scaling": 4.0}),
        Scenario("lmp", "lmp", {"lmp_scale": 2.0}),
        Scenario("bfl", "bfl", {"bfl_poison_frac": 1.0}),
        Scenario("dba", "dba", {"dba_poison_frac": 0.5, "dba_num_sub_triggers": 4}),
    ]
    scenarios = _filter_scenarios(scenarios, args.scenarios)
    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    records = []

    print("CONFIG", vars(args))
    print("SCENARIO DEFENSE reward clean backdoor attack_damage")
    for scenario in scenarios:
        for defense in defenses:
            config = _patched_config(base_config, {**scenario.patch, "defense_type": defense})
            runner = MinimalFLRunner(config)
            attack = create_attack(_attack_config(config, scenario.attack_type))
            summaries = runner.run_many_rounds(args.rounds, attack=attack, eval_every=1)
            last = summaries[-1]
            reward = _reward(last.clean_acc, last.backdoor_acc, targeted=scenario.attack_type in {"bfl", "dba"})
            damage = last.backdoor_acc if scenario.attack_type in {"bfl", "dba"} else 1.0 - last.clean_acc
            if scenario.attack_type == "clean":
                damage = float("nan")
            record = {
                "scenario": scenario.name,
                "attack_type": scenario.attack_type,
                "defense": defense,
                "reward": round(float(reward), 6),
                "clean_acc": round(float(last.clean_acc), 6),
                "backdoor_acc": round(float(last.backdoor_acc), 6),
                "attack_damage": (
                    round(float(damage), 6) if np.isfinite(damage) else None
                ),
                "rounds": int(args.rounds),
                "num_clients": int(args.num_clients),
                "num_attackers": int(config.fl.num_attackers),
                "subsample_rate": float(args.subsample_rate),
                "seed": int(args.seed),
            }
            records.append(record)
            print(
                scenario.name,
                defense,
                round(reward, 4),
                round(float(last.clean_acc), 4),
                round(float(last.backdoor_acc), 4),
                round(float(damage), 4) if np.isfinite(damage) else "nan",
            )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _filter_scenarios(scenarios: list[Scenario], selected: str) -> list[Scenario]:
    by_name = {scenario.name: scenario for scenario in scenarios}
    names = [item.strip() for item in selected.split(",") if item.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        supported = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}. Supported: {supported}")
    return [by_name[name] for name in names]


def _patched_config(base: SandboxConfig, patch: dict) -> SandboxConfig:
    values = base.to_flat_dict()
    values.update(patch)
    return SandboxConfig(**values)


def _attack_config(config: SandboxConfig, attack_type: str):
    values = config.to_flat_dict()
    values["type"] = attack_type
    return argparse.Namespace(**values)


def _reward(clean_acc: float, backdoor_acc: float, *, targeted: bool) -> float:
    if targeted:
        return float(clean_acc - backdoor_acc)
    return float(clean_acc)


if __name__ == "__main__":
    main()
