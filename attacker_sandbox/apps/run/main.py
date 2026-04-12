"""Unified experiment entry point for clean and attack sandbox runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATTACK_CHOICES = ["clean", "ipm", "lmp", "bfl", "dba", "rl", "brl"]


def parse_args(
    argv: Optional[list[str]] = None,
    *,
    default_attack_type: str = "clean",
    description: str = "Unified attacker sandbox experiment runner",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_attackers", type=int, default=None)
    parser.add_argument("--subsample_rate", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--parallel_clients", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_type", type=str, default=default_attack_type, choices=ATTACK_CHOICES)
    parser.add_argument("--ipm_scaling", type=float, default=2.0)
    parser.add_argument("--lmp_scale", type=float, default=2.0)
    parser.add_argument("--base_class", type=int, default=1)
    parser.add_argument("--target_class", type=int, default=7)
    parser.add_argument("--pattern_type", type=str, default="square")
    parser.add_argument("--bfl_poison_frac", type=float, default=1.0)
    parser.add_argument("--dba_poison_frac", type=float, default=0.5)
    parser.add_argument("--dba_num_sub_triggers", type=int, default=4)
    parser.add_argument("--attacker_action", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tb_dir", type=str, default="")
    return parser.parse_args(argv)


def _resolve_num_attackers(args: argparse.Namespace) -> int:
    if args.num_attackers is not None:
        return args.num_attackers
    return 0 if args.attack_type == "clean" else 2


def _default_output_dir(args: argparse.Namespace) -> str:
    if args.attack_type == "clean":
        return "attacker_sandbox/outputs/clean_benchmark"
    return f"attacker_sandbox/outputs/{args.attack_type}_demo"


def _default_tb_dir(args: argparse.Namespace) -> str:
    if args.attack_type == "clean":
        return "attacker_sandbox/runs/clean_benchmark"
    return f"attacker_sandbox/runs/{args.attack_type}_demo"


def _build_attack(args: argparse.Namespace):
    from attacker_sandbox.attacks import BFLAttack, BRLAttack, DBAAttack, IPMAttack, LMPAttack, RLAttack

    if args.attack_type == "clean":
        return None
    if args.attack_type == "ipm":
        return IPMAttack(scaling=args.ipm_scaling)
    if args.attack_type == "lmp":
        return LMPAttack(scale=args.lmp_scale)
    if args.attack_type == "bfl":
        return BFLAttack(poison_frac=args.bfl_poison_frac)
    if args.attack_type == "dba":
        return DBAAttack(
            num_sub_triggers=args.dba_num_sub_triggers,
            poison_frac=args.dba_poison_frac,
        )
    if args.attack_type == "rl":
        return RLAttack(default_action=tuple(args.attacker_action))
    if args.attack_type == "brl":
        return BRLAttack(default_action=tuple(args.attacker_action))
    raise ValueError(f"Unsupported attack type: {args.attack_type}")


def _build_config(args: argparse.Namespace) -> SandboxConfig:
    from attacker_sandbox.fl_runner import SandboxConfig

    return SandboxConfig(
        dataset=args.dataset,
        data_dir="data",
        device=args.device,
        seed=args.seed,
        num_clients=args.num_clients,
        num_attackers=_resolve_num_attackers(args),
        subsample_rate=args.subsample_rate,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        parallel_clients=args.parallel_clients,
        base_class=args.base_class,
        target_class=args.target_class,
        pattern_type=args.pattern_type,
        bfl_poison_frac=args.bfl_poison_frac,
        dba_poison_frac=args.dba_poison_frac,
        dba_num_sub_triggers=args.dba_num_sub_triggers,
        attacker_action=tuple(args.attacker_action),
    )


def _write_tensorboard(
    tb_dir: Path,
    attack_type: str,
    summaries,
    series: dict[str, list[float]],
    total_seconds: float,
    config_payload: dict[str, object],
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config/json", json.dumps(config_payload, indent=2), global_step=0)

    metric_prefix = "clean" if attack_type == "clean" else "attack"
    for summary in summaries:
        if not math.isnan(summary.clean_loss):
            writer.add_scalar(f"{metric_prefix}/loss", summary.clean_loss, summary.round_idx)
        if not math.isnan(summary.clean_acc):
            writer.add_scalar(f"{metric_prefix}/accuracy", summary.clean_acc, summary.round_idx)
        if not math.isnan(summary.backdoor_acc):
            writer.add_scalar(f"{metric_prefix}/backdoor_accuracy", summary.backdoor_acc, summary.round_idx)
        writer.add_scalar(f"{metric_prefix}/round_seconds", summary.round_seconds, summary.round_idx)
        writer.add_scalar(f"{metric_prefix}/num_sampled_clients", len(summary.sampled_clients), summary.round_idx)

    if attack_type != "clean":
        for round_idx, value in enumerate(series["mean_benign_norm"], start=1):
            writer.add_scalar("attack/mean_benign_norm", value, round_idx)
        for round_idx, value in enumerate(series["mean_malicious_norm"], start=1):
            writer.add_scalar("attack/mean_malicious_norm", value, round_idx)
        for round_idx, value in enumerate(series["mean_malicious_cosine"], start=1):
            writer.add_scalar("attack/mean_malicious_cosine", value, round_idx)

    writer.add_scalar(f"{metric_prefix}/total_seconds", total_seconds, 0)
    writer.flush()
    writer.close()


def _build_payload(
    args: argparse.Namespace,
    config: SandboxConfig,
    series: dict[str, list[float]],
    summaries,
    total_seconds: float,
) -> dict[str, object]:
    final_clean_acc = next((acc for acc in reversed(series["clean_acc"]) if not math.isnan(acc)), float("nan"))
    final_backdoor_acc = next((acc for acc in reversed(series["backdoor_acc"]) if not math.isnan(acc)), float("nan"))
    config_payload = {
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
        "eval_every": args.eval_every,
        "base_class": config.base_class,
        "target_class": config.target_class,
        "pattern_type": config.pattern_type,
        "attack_type": args.attack_type,
        "ipm_scaling": args.ipm_scaling,
        "lmp_scale": args.lmp_scale,
        "bfl_poison_frac": args.bfl_poison_frac,
        "dba_poison_frac": args.dba_poison_frac,
        "dba_num_sub_triggers": args.dba_num_sub_triggers,
        "attacker_action": list(args.attacker_action),
        "rounds": args.rounds,
    }
    return {
        "config": config_payload,
        "attack_type": args.attack_type,
        "total_seconds": total_seconds,
        "series": series,
        "rounds": [
            {
                "round_idx": summary.round_idx,
                "sampled_clients": summary.sampled_clients,
                "selected_attackers": summary.selected_attackers,
                "clean_loss": summary.clean_loss,
                "clean_acc": summary.clean_acc,
                "backdoor_acc": summary.backdoor_acc,
                "round_seconds": summary.round_seconds,
                "mean_benign_norm": series["mean_benign_norm"][summary.round_idx - 1],
                "mean_malicious_norm": series["mean_malicious_norm"][summary.round_idx - 1],
                "mean_malicious_cosine": series["mean_malicious_cosine"][summary.round_idx - 1],
                "evaluated": not math.isnan(summary.clean_acc),
            }
            for summary in summaries
        ],
        "final": {
            "clean_acc": final_clean_acc,
            "backdoor_acc": final_backdoor_acc,
            "mean_benign_norm": series["mean_benign_norm"][-1] if series["mean_benign_norm"] else float("nan"),
            "mean_malicious_norm": series["mean_malicious_norm"][-1] if series["mean_malicious_norm"] else float("nan"),
            "mean_malicious_cosine": series["mean_malicious_cosine"][-1] if series["mean_malicious_cosine"] else float("nan"),
        },
    }


def main(
    argv: Optional[list[str]] = None,
    *,
    default_attack_type: str = "clean",
    description: str = "Unified attacker sandbox experiment runner",
) -> None:
    args = parse_args(argv, default_attack_type=default_attack_type, description=description)
    from attacker_sandbox.fl_runner import MinimalFLRunner, client_metrics_to_rows, summaries_to_dict

    config = _build_config(args)
    attack = _build_attack(args)

    output_dir = Path(args.output_dir or _default_output_dir(args))
    tb_dir = Path(args.tb_dir or _default_tb_dir(args))
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    runner = MinimalFLRunner(config)
    start = time.time()
    summaries = runner.run_many_rounds(
        args.rounds,
        attack=attack,
        show_progress=True,
        progress_desc=f"{args.attack_type} ({config.dataset})",
        eval_every=args.eval_every,
        attacker_action=None if attack is None else tuple(args.attacker_action),
    )
    total_seconds = time.time() - start
    series = summaries_to_dict(summaries)
    payload = _build_payload(args, config, series, summaries, total_seconds)
    config_payload = payload["config"]

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
        writer_csv.writerows(client_metrics_to_rows(summaries))

    _write_tensorboard(tb_dir, args.attack_type, summaries, series, total_seconds, config_payload)

    print("Sandbox run finished.")
    print(f"Mode: {args.attack_type}")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard dir: {tb_dir}")
    print(f"Summary file: {output_dir / 'summary.json'}")
    print(f"Client metrics table: {output_dir / 'client_metrics.csv'}")
    if args.attack_type == "clean":
        print(
            "Run postprocess with: "
            f"python attacker_sandbox/apps/postprocess/postprocess_clean.py --input_dir {output_dir} --tb_dir {tb_dir}"
        )
    else:
        suggested_clean_dir = Path(
            f"attacker_sandbox/outputs/{config.dataset}_clean_{config.num_clients}c_{args.rounds}r"
        )
        print(
            "Run postprocess with: "
            "python attacker_sandbox/apps/postprocess/postprocess_sandbox.py "
            f"--clean_input_dir {suggested_clean_dir} "
            f"--attack_input_dir {output_dir} --tb_dir {tb_dir}_compare"
        )
    print(f"Runtime device: {runner.device}")
    print(f"Total seconds: {total_seconds:.3f}")
    print(f"Final clean acc: {payload['final']['clean_acc']:.4f}")
    print(f"Final backdoor acc: {payload['final']['backdoor_acc']:.4f}")
    if args.attack_type != "clean":
        print(f"Final mean malicious norm: {payload['final']['mean_malicious_norm']:.4f}")
        print(f"Final mean malicious cosine: {payload['final']['mean_malicious_cosine']:.4f}")


if __name__ == "__main__":
    main()
