"""Smoke response-curve runner for Meta-SG M1/M2 co-learning."""
from __future__ import annotations

import argparse

from meta_sg.co_learning.milestones import sweep_fixed_defenders
from meta_sg.learning.config import TD3Config
from meta_sg.simulation.stub import StubCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.5, 2.5, 4.5])
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--br-updates", type=int, default=0)
    parser.add_argument("--br-episodes", type=int, default=1)
    parser.add_argument("--eval-after-updates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    td3_config = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        warmup_steps=0,
    )

    def coordinator_factory():
        return StubCoordinator(num_clients=6, num_attackers=2, subsample_rate=1.0, seed=int(args.seed))

    records = sweep_fixed_defenders(
        coordinator_factory=coordinator_factory,
        radii=args.radii,
        trim_ratio=float(args.trim_ratio),
        horizon=int(args.horizon),
        td3_config=td3_config,
        seed=int(args.seed),
        br_updates=int(args.br_updates),
        br_episodes=int(args.br_episodes),
        eval_after_updates=bool(args.eval_after_updates),
    )
    print(
        "clip_radius,trim_ratio,mean_epsilon,mean_local_steps,"
        "mean_attacker_reward,mean_survival,mean_stealth_cost,attacker_buffer_size"
    )
    for record in records:
        print(
            f"{record.clip_radius:.6g},{record.trim_ratio:.6g},"
            f"{record.mean_epsilon:.6g},{record.mean_local_steps:.6g},"
            f"{record.mean_attacker_reward:.6g},{record.mean_survival:.6g},"
            f"{record.mean_stealth_cost:.6g},"
            f"{record.attacker_buffer_size}"
        )


if __name__ == "__main__":
    main()
