"""Run larger M1/M2 co-learning response experiments with TensorBoard logs."""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meta_sg.co_learning.milestones import FixedClippedMedianPolicy, run_m1_episode
from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.stub import StubCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["stub", "fl_sandbox"], default="stub")
    parser.add_argument("--log-root", default="runs/meta_sg_colearning_m1m2")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--radii", type=float, nargs="+", default=[0.5, 1.5, 2.5, 3.5, 4.5])
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--br-episodes", type=int, default=5)
    parser.add_argument("--br-updates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=5000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--client-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def total_training_rounds(*, num_radii: int, outer_iters: int, horizon: int, br_episodes: int) -> int:
    return int(num_radii) * int(outer_iters) * int(horizon) * int(br_episodes)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_root) / f"{args.backend}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    csv_path = log_dir / "m1_m2_metrics.csv"

    td3_config = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.exploration_noise),
    )
    total_rounds = total_training_rounds(
        num_radii=len(args.radii),
        outer_iters=args.outer_iters,
        horizon=args.horizon,
        br_episodes=args.br_episodes,
    )
    _write_hparams(writer, args, total_rounds)

    print(f"LOG_DIR {log_dir}", flush=True)
    print(f"CSV {csv_path}", flush=True)
    print(f"TOTAL_TRAINING_FL_ROUNDS {total_rounds}", flush=True)
    print(
        "iter,radius,epsilon,local_steps,reward,survival,stealth,buffer,updates,eval_transitions",
        flush=True,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "outer_iter",
            "radius",
            "trim_ratio",
            "mean_epsilon",
            "mean_local_steps",
            "mean_attacker_reward",
            "mean_survival",
            "mean_stealth_cost",
            "attacker_buffer_size",
            "attacker_updates",
            "eval_transitions",
        ]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()

        for radius_idx, radius in enumerate(args.radii):
            coordinator = _make_coordinator(args, seed=args.seed + radius_idx)
            obs_dim = obs_dim_for(coordinator.spec.empty_weights())
            attacker = TD3Agent(obs_dim=obs_dim, act_dim=2, config=td3_config)
            defender = FixedClippedMedianPolicy(clip_radius=float(radius), trim_ratio=float(args.trim_ratio))
            for outer_iter in range(int(args.outer_iters)):
                result = run_m1_episode(
                    coordinator=_make_coordinator(args, seed=args.seed + radius_idx * 1000 + outer_iter),
                    attacker=attacker,
                    defender=defender,
                    horizon=int(args.horizon),
                    seed=args.seed + radius_idx * 1000 + outer_iter,
                    td3_config=td3_config,
                    br_updates=int(args.br_updates),
                    br_episodes=int(args.br_episodes),
                    eval_after_updates=True,
                )
                step = outer_iter
                tag_prefix = f"radius_{float(radius):.2f}".replace(".", "_")
                _log_result(writer, tag_prefix, step, result)
                row = {
                    "outer_iter": outer_iter,
                    "radius": float(radius),
                    "trim_ratio": float(args.trim_ratio),
                    "mean_epsilon": result.mean_epsilon,
                    "mean_local_steps": result.mean_local_steps,
                    "mean_attacker_reward": result.mean_attacker_reward,
                    "mean_survival": result.mean_survival,
                    "mean_stealth_cost": result.mean_stealth_cost,
                    "attacker_buffer_size": result.attacker_buffer_size,
                    "attacker_updates": result.attacker_updates,
                    "eval_transitions": result.eval_transitions_collected,
                }
                writer_csv.writerow(row)
                handle.flush()
                print(
                    f"{outer_iter},{radius:.3g},{result.mean_epsilon:.4g},"
                    f"{result.mean_local_steps:.4g},{result.mean_attacker_reward:.6g},"
                    f"{result.mean_survival:.6g},{result.mean_stealth_cost:.6g},"
                    f"{result.attacker_buffer_size},{result.attacker_updates},"
                    f"{result.eval_transitions_collected}",
                    flush=True,
                )
            writer.flush()
    writer.close()


def _make_coordinator(args: argparse.Namespace, *, seed: int):
    if args.backend == "stub":
        return StubCoordinator(
            num_clients=int(args.num_clients),
            num_attackers=int(args.num_attackers),
            subsample_rate=1.0,
            seed=int(seed),
        )
    config = SandboxConfig(
        dataset="mnist",
        device=args.device,
        seed=int(seed),
        num_clients=int(args.num_clients),
        num_attackers=int(args.num_attackers),
        subsample_rate=1.0,
        local_epochs=1,
        batch_size=int(args.batch_size_fl),
        eval_batch_size=int(args.eval_batch_size),
        max_client_samples_per_client=int(args.client_samples),
        max_eval_samples=int(args.eval_samples),
        parallel_clients=1,
        fltrust_root_size=0,
    )
    return FLSandboxCoordinatorAdapter(config)


def _write_hparams(writer: SummaryWriter, args: argparse.Namespace, total_rounds: int) -> None:
    writer.add_text("config/backend", args.backend, 0)
    writer.add_text("config/radii", ",".join(str(radius) for radius in args.radii), 0)
    writer.add_scalar("config/total_training_fl_rounds", float(total_rounds), 0)
    writer.add_scalar("config/outer_iters", float(args.outer_iters), 0)
    writer.add_scalar("config/horizon", float(args.horizon), 0)
    writer.add_scalar("config/br_episodes", float(args.br_episodes), 0)
    writer.add_scalar("config/br_updates", float(args.br_updates), 0)
    writer.add_scalar("config/num_clients", float(args.num_clients), 0)
    writer.add_scalar("config/num_attackers", float(args.num_attackers), 0)


def _log_result(writer: SummaryWriter, prefix: str, step: int, result) -> None:
    scalars = {
        "mean_epsilon": result.mean_epsilon,
        "mean_local_steps": result.mean_local_steps,
        "mean_attacker_reward": result.mean_attacker_reward,
        "mean_survival": result.mean_survival,
        "mean_stealth_cost": result.mean_stealth_cost,
        "attacker_buffer_size": result.attacker_buffer_size,
        "attacker_updates": result.attacker_updates,
        "eval_transitions": result.eval_transitions_collected,
    }
    for name, value in scalars.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), int(step))
    losses = result.attacker_update_losses or []
    if losses:
        for key in sorted({key for loss in losses for key in loss}):
            values = [float(loss[key]) for loss in losses if key in loss and np.isfinite(loss[key])]
            if values:
                writer.add_scalar(f"{prefix}/loss/{key}", float(np.mean(values)), int(step))


if __name__ == "__main__":
    main()
