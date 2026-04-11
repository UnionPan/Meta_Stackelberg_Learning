"""Clean FL benchmark with saved plots and per-round records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_sandbox.fl_runner import (
    MinimalFLRunner,
    SandboxConfig,
    client_metrics_to_rows,
    summaries_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark clean FL in the attacker sandbox")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_clients", type=int, default=6)
    parser.add_argument("--subsample_rate", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--parallel_clients", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attacker_sandbox/outputs/clean_benchmark",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default="attacker_sandbox/runs/clean_benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SandboxConfig(
        dataset=args.dataset,
        data_dir="data",
        device=args.device,
        seed=args.seed,
        num_clients=args.num_clients,
        num_attackers=0,
        subsample_rate=args.subsample_rate,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        parallel_clients=args.parallel_clients,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    runner = MinimalFLRunner(config)
    summaries = runner.run_many_rounds(
        args.rounds,
        attack=None,
        show_progress=True,
        progress_desc=f"Clean FL ({config.dataset})",
        eval_every=args.eval_every,
    )
    total_seconds = time.time() - start
    series = summaries_to_dict(summaries)
    writer = SummaryWriter(log_dir=str(tb_dir))

    writer.add_text(
        "config/json",
        json.dumps(
            {
                "dataset": config.dataset,
                "data_dir": config.data_dir,
                "seed": config.seed,
                "num_clients": config.num_clients,
                "subsample_rate": config.subsample_rate,
                "local_epochs": config.local_epochs,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "eval_batch_size": config.eval_batch_size,
                "num_workers": config.num_workers,
                "parallel_clients": config.parallel_clients,
                "eval_every": args.eval_every,
            },
            indent=2,
        ),
        global_step=0,
    )

    for summary in summaries:
        if not math.isnan(summary.clean_loss):
            writer.add_scalar("clean/loss", summary.clean_loss, summary.round_idx)
        if not math.isnan(summary.clean_acc):
            writer.add_scalar("clean/accuracy", summary.clean_acc, summary.round_idx)
        writer.add_scalar("clean/round_seconds", summary.round_seconds, summary.round_idx)
        writer.add_scalar(
            "clean/num_sampled_clients",
            len(summary.sampled_clients),
            summary.round_idx,
        )

    writer.add_scalar("clean/total_seconds", total_seconds, 0)

    payload = {
        "config": {
            "dataset": config.dataset,
            "data_dir": config.data_dir,
            "seed": config.seed,
            "num_clients": config.num_clients,
            "subsample_rate": config.subsample_rate,
            "local_epochs": config.local_epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "num_workers": config.num_workers,
            "parallel_clients": config.parallel_clients,
            "eval_every": args.eval_every,
        },
        "total_seconds": total_seconds,
        "rounds": [
            {
                "round_idx": summary.round_idx,
                "sampled_clients": summary.sampled_clients,
                "clean_loss": summary.clean_loss,
                "clean_acc": summary.clean_acc,
                "evaluated": not math.isnan(summary.clean_acc),
                "round_seconds": summary.round_seconds,
            }
            for summary in summaries
        ],
    }
    client_metric_rows = client_metrics_to_rows(summaries)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with (output_dir / "client_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer_csv = csv.DictWriter(
            fh,
            fieldnames=[
                "round_idx",
                "client_id",
                "selected",
                "is_attacker",
                "train_loss",
                "train_acc",
                "update_norm",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(client_metric_rows)
    writer.flush()
    writer.close()

    print("Clean benchmark finished.")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard dir: {tb_dir}")
    print(f"Client metrics table: {output_dir / 'client_metrics.csv'}")
    print("Run postprocess with: "
          f"python attacker_sandbox/postprocess_clean.py --input_dir {output_dir} --tb_dir {tb_dir}")
    print(f"Runtime device: {runner.device}")
    print(f"Total seconds: {total_seconds:.3f}")
    last_eval_acc = next((acc for acc in reversed(series["clean_acc"]) if not math.isnan(acc)), float("nan"))
    print(f"Final clean acc: {last_eval_acc:.4f}")


if __name__ == "__main__":
    main()
