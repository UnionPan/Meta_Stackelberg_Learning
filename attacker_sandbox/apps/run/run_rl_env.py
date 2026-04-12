"""Smoke-test entry point for the single-agent attacker RL environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_sandbox.fl_runner import SandboxConfig
from attacker_sandbox.rl_env import AttackerRLEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal rollout in attacker_sandbox.rl_env")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--attack_type", type=str, default="rl", choices=["ipm", "lmp", "bfl", "dba", "rl", "brl"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_attackers", type=int, default=2)
    parser.add_argument("--subsample_rate", type=float, default=1.0)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_class", type=int, default=1)
    parser.add_argument("--target_class", type=int, default=7)
    parser.add_argument("--pattern_type", type=str, default="square")
    parser.add_argument("--attacker_action", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output_path", type=str, default="attacker_sandbox/outputs/rl_env_rollout.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SandboxConfig(
        dataset=args.dataset,
        device=args.device,
        seed=args.seed,
        num_clients=args.num_clients,
        num_attackers=args.num_attackers,
        subsample_rate=args.subsample_rate,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        base_class=args.base_class,
        target_class=args.target_class,
        pattern_type=args.pattern_type,
        attacker_action=tuple(args.attacker_action),
    )
    env = AttackerRLEnv(config=config, attack_type=args.attack_type, max_rounds=args.rounds)
    obs, info = env.reset(seed=args.seed)
    rollout = {
        "initial_obs_shape": list(obs.shape),
        "reset_info": info,
        "steps": [],
    }
    action = np.asarray(args.attacker_action, dtype=np.float32)
    done = False
    while not done:
        obs, reward, terminated, truncated, step_info = env.step(action)
        rollout["steps"].append(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": step_info,
            }
        )
        done = terminated or truncated
    env.close()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rollout, fh, indent=2)

    print(f"Rollout saved to: {output_path}")
    if rollout["steps"]:
        final = rollout["steps"][-1]["info"]
        print(f"Final clean_acc: {final['clean_acc']:.4f}")
        print(f"Final backdoor_acc: {final['backdoor_acc']:.4f}")


if __name__ == "__main__":
    main()
