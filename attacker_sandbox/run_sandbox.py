"""Entry point for standalone attacker demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_sandbox.attacks import IPMAttack
from attacker_sandbox.fl_runner import MinimalFLRunner, SandboxConfig, summaries_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone attacker sandbox")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_attackers", type=int, default=2)
    parser.add_argument("--subsample_rate", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--parallel_clients", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ipm_scaling", type=float, default=2.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attacker_sandbox/outputs/ipm_demo",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default="attacker_sandbox/runs/ipm_demo",
    )
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
        parallel_clients=args.parallel_clients,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.tb_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)

    clean_runner = MinimalFLRunner(config)
    clean_summaries = clean_runner.run_many_rounds(
        args.rounds,
        attack=None,
        show_progress=True,
        progress_desc=f"Clean FL ({config.dataset})",
    )

    ipm_runner = MinimalFLRunner(config)
    ipm_summaries = ipm_runner.run_many_rounds(
        args.rounds,
        attack=IPMAttack(scaling=args.ipm_scaling),
        show_progress=True,
        progress_desc=f"IPM attack ({config.dataset})",
    )

    clean_series = summaries_to_dict(clean_summaries)
    ipm_series = summaries_to_dict(ipm_summaries)
    writer = SummaryWriter(log_dir=str(tb_dir))
    for round_idx, clean_acc in enumerate(clean_series["clean_acc"], start=1):
        clean_loss = clean_series["clean_loss"][round_idx - 1]
        if not (clean_loss != clean_loss):
            writer.add_scalar("clean/loss", clean_loss, round_idx)
        if not (clean_acc != clean_acc):
            writer.add_scalar("clean/accuracy", clean_acc, round_idx)
    for round_idx, ipm_acc in enumerate(ipm_series["clean_acc"], start=1):
        ipm_loss = ipm_series["clean_loss"][round_idx - 1]
        if not (ipm_loss != ipm_loss):
            writer.add_scalar("ipm/loss", ipm_loss, round_idx)
        if not (ipm_acc != ipm_acc):
            writer.add_scalar("ipm/accuracy", ipm_acc, round_idx)
    writer.flush()
    writer.close()

    payload = {
        "config": {
            "dataset": config.dataset,
            "data_dir": config.data_dir,
            "device": args.device,
            "seed": config.seed,
            "num_clients": config.num_clients,
            "num_attackers": config.num_attackers,
            "subsample_rate": config.subsample_rate,
            "local_epochs": config.local_epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "num_workers": config.num_workers,
            "parallel_clients": config.parallel_clients,
            "ipm_scaling": args.ipm_scaling,
            "rounds": args.rounds,
        },
        "runs": {
            "clean": clean_series,
            "ipm": ipm_series,
        },
        "final": {
            "clean_acc": clean_series["clean_acc"][-1],
            "ipm_acc": ipm_series["clean_acc"][-1],
            "ipm_mean_malicious_norm": ipm_series["mean_malicious_norm"][-1],
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Sandbox run finished.")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard dir: {tb_dir}")
    print(f"Summary file: {output_dir / 'summary.json'}")
    print(
        "Run postprocess with: "
        f"python attacker_sandbox/postprocess_sandbox.py --input_dir {output_dir} --tb_dir {tb_dir}"
    )
    print(f"Runtime device: {clean_runner.device}")
    print(f"Clean final acc: {clean_series['clean_acc'][-1]:.4f}")
    print(f"IPM final acc:   {ipm_series['clean_acc'][-1]:.4f}")
    print(f"IPM mean malicious norm: {ipm_series['mean_malicious_norm'][-1]:.4f}")


if __name__ == "__main__":
    main()
