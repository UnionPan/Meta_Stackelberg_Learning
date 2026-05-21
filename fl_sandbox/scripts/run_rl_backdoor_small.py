"""Small RL-backdoor benchmark against BFL and DBA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fl_sandbox.attacks import create_attack
from fl_sandbox.attacks.rl_backdoor.policy import EliteBackdoorPolicy
from fl_sandbox.config import RunConfig
from fl_sandbox.federation.runner import MinimalFLRunner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--train-rounds", type=int, default=6)
    parser.add_argument("--dataset", default="mnist", choices=("mnist", "fmnist", "cifar10"))
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--subsample-rate", type=float, default=1.0)
    parser.add_argument("--client-samples", type=int, default=16, help="<=0 uses the full local client split")
    parser.add_argument("--eval-samples", type=int, default=128, help="<=0 uses the full target-class eval split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--bfl-poison-frac", type=float, default=1.0)
    parser.add_argument("--dba-poison-frac", type=float, default=0.5)
    parser.add_argument("--base-class", type=int, default=1)
    parser.add_argument("--target-class", type=int, default=7)
    parser.add_argument("--pattern-type", default="square")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("fl_sandbox/outputs/rl_backdoor_small"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    action_candidates = [
        np.array([1.0, -0.4, -1.0, -0.2], dtype=np.float32),
        np.array([1.0, 0.0, -0.6, 0.0], dtype=np.float32),
        np.array([1.0, 0.3, -0.2, 0.2], dtype=np.float32),
        np.array([0.5, 0.0, -0.6, 0.5], dtype=np.float32),
    ]
    policy = EliteBackdoorPolicy()
    train_rows = []
    for idx, action in enumerate(action_candidates):
        summary = _run_attack("rl_backdoor", args, rounds=args.train_rounds, action=action, seed=args.seed + idx)
        reward = _reward_from_summary(summary)
        policy.observe(action=action, reward=reward)
        train_rows.append({"action": action.tolist(), "reward": reward, **summary})

    policy_path = args.output_dir / "rl_backdoor_elite_policy.json"
    policy.save(policy_path)

    eval_rows = {
        "bfl": _run_attack("bfl", args, rounds=args.rounds, action=None, seed=args.seed + 100),
        "dba": _run_attack("dba", args, rounds=args.rounds, action=None, seed=args.seed + 200),
        "rl_backdoor": _run_attack("rl_backdoor", args, rounds=args.rounds, action=policy.act(), seed=args.seed + 300),
    }
    payload = {
        "policy_path": str(policy_path),
        "best_action": policy.act().tolist(),
        "best_reward": policy.best_reward,
        "train": train_rows,
        "eval": eval_rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("SUMMARY", summary_path)
    print("POLICY", policy_path)
    for name, row in eval_rows.items():
        print(
            "RESULT",
            name,
            "clean=",
            round(row["final_clean_acc"], 4),
            "asr=",
            round(row["final_backdoor_acc"], 4),
            "last_asr=",
            round(row["last_backdoor_acc"], 4),
        )


def _run_attack(attack_type: str, args: argparse.Namespace, *, rounds: int, action, seed: int) -> dict[str, float]:
    cfg = RunConfig.from_flat_dict(
        {
            "dataset": args.dataset,
            "attack_type": attack_type,
            "defense_type": "fedavg",
            "rounds": rounds,
            "device": args.device,
            "num_clients": args.num_clients,
            "num_attackers": args.num_attackers,
            "subsample_rate": args.subsample_rate,
            "local_epochs": 1,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_client_samples_per_client": args.client_samples if args.client_samples > 0 else None,
            "max_eval_samples": args.eval_samples if args.eval_samples > 0 else None,
            "seed": seed,
            "bfl_poison_frac": args.bfl_poison_frac,
            "dba_poison_frac": args.dba_poison_frac,
            "dba_num_sub_triggers": 4,
            "base_class": args.base_class,
            "target_class": args.target_class,
            "pattern_type": args.pattern_type,
        }
    )
    runner = MinimalFLRunner(cfg)
    attack = create_attack(cfg.attacker)
    summaries = []
    for round_idx in range(1, rounds + 1):
        summaries.append(runner.run_round(round_idx, attack=attack, evaluate=True, attacker_action=action))
    last = summaries[-1]
    clean_values = [float(s.clean_acc) for s in summaries if math.isfinite(float(s.clean_acc))]
    backdoor_values = [float(s.backdoor_acc) for s in summaries if math.isfinite(float(s.backdoor_acc))]
    return {
        "final_clean_acc": float(last.clean_acc),
        "final_backdoor_acc": float(last.backdoor_acc),
        "last_clean_acc": clean_values[-1] if clean_values else float("nan"),
        "last_backdoor_acc": backdoor_values[-1] if backdoor_values else float("nan"),
        "mean_backdoor_acc": float(np.mean(backdoor_values)) if backdoor_values else float("nan"),
    }


def _reward_from_summary(summary: dict[str, float]) -> float:
    asr = float(summary["final_backdoor_acc"])
    clean = float(summary["final_clean_acc"])
    return asr - max(0.0, 0.75 - clean)


if __name__ == "__main__":
    main(sys.argv[1:])
