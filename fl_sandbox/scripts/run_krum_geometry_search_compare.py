"""Compare clean Krum and Krum geometry-search attackers with TensorBoard logging.

Example:
    .venv/bin/python fl_sandbox/scripts/run_krum_geometry_search_compare.py --rounds 30

TensorBoard:
    tensorboard --logdir fl_sandbox/runs/krum_geometry_search_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.tensorboard import SummaryWriter

from fl_sandbox.attacks.krum_geometry_search import KrumGeometrySearchAttack, KrumGeometrySearchConfig
from fl_sandbox.federation.runner import MinimalFLRunner, SandboxConfig


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(values) if values else float("nan")


def _tail_summary(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_clean_acc": _mean(row["clean_acc"] for row in rows),
        f"{prefix}_clean_loss": _mean(row["clean_loss"] for row in rows),
        f"{prefix}_selected_rate": _mean(row.get("selected", 0.0) for row in rows),
        f"{prefix}_alpha": _mean(row.get("alpha", 0.0) for row in rows),
        f"{prefix}_score_ratio": _mean(row.get("score_ratio", 0.0) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=3)
    parser.add_argument("--subsample-rate", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-client-samples", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--parallel-clients", type=int, default=1)
    parser.add_argument("--split-mode", default="iid", choices=["iid", "noniid", "paper_q"])
    parser.add_argument("--noniid-q", type=float, default=0.5)
    parser.add_argument("--max-alpha", type=float, default=5.0)
    parser.add_argument("--geometry-search-steps", dest="geometry_search_steps", type=int, default=24)
    parser.add_argument("--ascent-epochs", type=int, default=0)
    parser.add_argument("--attacks", nargs="+", default=["clean", "krum_geometry_search"], choices=["clean", "krum_geometry_search"])
    parser.add_argument("--tb-root", default="fl_sandbox/runs/krum_geometry_search_compare")
    parser.add_argument("--out-root", default="fl_sandbox/outputs/krum_geometry_search_compare")
    return parser.parse_args()


def make_runner(args: argparse.Namespace) -> MinimalFLRunner:
    cfg = SandboxConfig(
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
        max_client_samples_per_client=args.max_client_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=0,
        parallel_clients=args.parallel_clients,
        split_mode=args.split_mode,
        noniid_q=args.noniid_q,
        defense_type="krum",
        krum_attackers=args.num_attackers,
    )
    return MinimalFLRunner(cfg)


def make_attack(args: argparse.Namespace, attack_type: str):
    if attack_type == "clean":
        return None
    if attack_type == "krum_geometry_search":
        return KrumGeometrySearchAttack(
            KrumGeometrySearchConfig(
                max_alpha=args.max_alpha,
                search_steps=args.geometry_search_steps,
                ascent_epochs=args.ascent_epochs,
            )
        )
    raise ValueError(f"Unsupported attack_type: {attack_type}")


def metric_alias(metrics: dict[str, float], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in metrics and _finite(metrics[name]):
            return float(metrics[name])
    return float(default)


def run_one(args: argparse.Namespace, attack_type: str, root_name: str) -> dict[str, object]:
    runner = make_runner(args)
    attack = make_attack(args, attack_type)
    tb_dir = Path(args.tb_root) / root_name / attack_type
    out_dir = Path(args.out_root) / root_name / attack_type
    tb_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    writer.add_text("config/json", json.dumps(vars(args), indent=2), global_step=0)

    rows: list[dict[str, float]] = []
    start = time.time()
    csv_path = out_dir / "round_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "round",
            "clean_loss",
            "clean_acc",
            "selected_attackers",
            "selected",
            "rank",
            "score_ratio",
            "alpha",
            "loss_delta",
            "round_seconds",
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for round_idx in range(1, args.rounds + 1):
            should_evaluate = args.eval_every <= 1 or round_idx % args.eval_every == 0 or round_idx == args.rounds
            summary = runner.run_round(round_idx, attack=attack, evaluate=should_evaluate)
            metrics = summary.attack_metrics or {}
            row = {
                "round": float(round_idx),
                "clean_loss": float(summary.clean_loss),
                "clean_acc": float(summary.clean_acc),
                "selected_attackers": float(len(summary.selected_attackers)),
                "selected": metric_alias(metrics, "krum_geometry_search_selected"),
                "rank": metric_alias(metrics, "krum_geometry_search_best_malicious_rank"),
                "score_ratio": metric_alias(metrics, "krum_geometry_search_score_ratio"),
                "alpha": metric_alias(metrics, "krum_attack_alpha"),
                "loss_delta": metric_alias(metrics, "krum_geometry_search_clean_loss_delta"),
                "round_seconds": float(summary.round_seconds),
            }
            rows.append(row)
            csv_writer.writerow(row)
            f.flush()

            if math.isfinite(float(summary.clean_loss)):
                writer.add_scalar("metrics/loss", summary.clean_loss, round_idx)
            if math.isfinite(float(summary.clean_acc)):
                writer.add_scalar("metrics/accuracy", summary.clean_acc, round_idx)
            writer.add_scalar("metrics/num_selected_attackers", len(summary.selected_attackers), round_idx)
            writer.add_scalar("metrics/round_seconds", summary.round_seconds, round_idx)
            writer.add_scalar("attack_compare/selected", row["selected"], round_idx)
            writer.add_scalar("attack_compare/rank", row["rank"], round_idx)
            writer.add_scalar("attack_compare/score_ratio", row["score_ratio"], round_idx)
            writer.add_scalar("attack_compare/alpha", row["alpha"], round_idx)
            writer.add_scalar("attack_compare/loss_delta", row["loss_delta"], round_idx)
            for key, value in metrics.items():
                if _finite(value):
                    writer.add_scalar(f"attack_only/{key}", float(value), round_idx)
            writer.flush()

            if round_idx <= 3 or round_idx == args.rounds or round_idx % 10 == 0:
                print(
                    json.dumps(
                        {
                            "attack": attack_type,
                            "round": round_idx,
                            "acc": round(row["clean_acc"], 4),
                            "loss": round(row["clean_loss"], 4),
                            "selected": round(row["selected"], 3),
                            "alpha": round(row["alpha"], 4),
                            "sec": round(time.time() - start, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    tail = rows[-min(10, len(rows)) :]
    result = {
        "attack_type": attack_type,
        "tb_dir": str(tb_dir),
        "csv_path": str(csv_path),
        "rounds": args.rounds,
        "seconds": time.time() - start,
        "final_clean_acc": rows[-1]["clean_acc"] if rows else float("nan"),
        "final_clean_loss": rows[-1]["clean_loss"] if rows else float("nan"),
        **_tail_summary(tail, "tail"),
    }
    writer.add_scalar("final/clean_acc", float(result["final_clean_acc"]), args.rounds)
    writer.add_scalar("final/clean_loss", float(result["final_clean_loss"]), args.rounds)
    writer.add_scalar("final/tail_selected_rate", float(result["tail_selected_rate"]), args.rounds)
    writer.add_scalar("final/tail_alpha", float(result["tail_alpha"]), args.rounds)
    writer.close()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    root_name = args.run_name or (
        f"{args.dataset}_krum_{args.rounds}r_{args.num_clients}c_"
        f"{args.num_attackers}a_seed{args.seed}"
    )
    all_results = []
    for attack_type in args.attacks:
        all_results.append(run_one(args, attack_type, root_name))

    compare_dir = Path(args.out_root) / root_name
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "comparison_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "root": root_name, "results": all_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
