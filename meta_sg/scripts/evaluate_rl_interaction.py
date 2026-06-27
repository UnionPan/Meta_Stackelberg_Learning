"""Evaluate learned RL defender against frozen-policy RL attacker and plot interactions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meta_sg.scripts.formal_eval_paper3d_defender import (
    evaluate_learned_from_warmup,
    load_defender,
    run_clean_warmup,
)
from meta_sg.scripts.run_stackelberg_defender_td3 import (
    resolve_attacker_checkpoint,
    resolve_distribution_dir,
)
from meta_sg.stackelberg.metrics import write_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defender-checkpoint", required=True)
    parser.add_argument("--attacker-checkpoint", default="")
    parser.add_argument("--rl-distribution-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="runs/stackelberg_paper3d_interaction_eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--eval-horizon", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--client-samples", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--exploration-noise", type=float, default=0.0)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=30.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.45)
    parser.add_argument("--neuroclip-eps-min", type=float, default=2.0)
    parser.add_argument("--neuroclip-eps-max", type=float, default=10.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    attacker_checkpoint = resolve_attacker_checkpoint(args.attacker_checkpoint)
    distribution_dir = resolve_distribution_dir(args.rl_distribution_dir, attacker_checkpoint)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / f"rl_interaction_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_weights, warmup_metrics = run_clean_warmup(args, output_dir=output_dir)
    defender = load_defender(args, attacker_checkpoint=attacker_checkpoint, distribution_dir=distribution_dir)
    rows = evaluate_learned_from_warmup(
        args,
        defender,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        warmup_weights=warmup_weights,
    )
    csv_path = output_dir / "rl_interaction_eval.csv"
    write_rows(csv_path, rows)
    summary = summarize_interaction(rows, warmup_metrics=warmup_metrics)
    summary.update(
        {
            "defender_checkpoint": str(Path(args.defender_checkpoint).expanduser()),
            "attacker_checkpoint": str(attacker_checkpoint),
            "rl_distribution_dir": str(distribution_dir),
            "steps": len(rows),
            "warmup_rounds": int(args.warmup_rounds),
            "eval_episodes": int(args.eval_episodes),
            "eval_horizon": int(args.eval_horizon or args.horizon),
            "csv": str(csv_path),
        }
    )
    (output_dir / "rl_interaction_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    plot_path = plot_interaction(rows, output_dir)
    print(
        "RL_INTERACTION_EVAL_RESULT",
        f"steps={len(rows)}",
        f"defender_alpha_std={summary['defender']['alpha']['std']:.6g}",
        f"attacker_gamma_std={summary['attacker']['gamma']['std']:.6g}",
        f"csv={csv_path}",
        f"plot={plot_path}",
        flush=True,
    )


def summarize_interaction(rows: list[dict], *, warmup_metrics: dict[str, float]) -> dict:
    return {
        "warmup_metrics": warmup_metrics,
        "performance": {
            "post_clean_acc": _stats(rows, "post_clean_acc"),
            "post_clean_loss": _stats(rows, "post_clean_loss"),
            "mean_malicious_norm": _stats(rows, "mean_malicious_norm"),
        },
        "defender": {
            "alpha": _stats(rows, "alpha"),
            "beta": _stats(rows, "beta"),
            "epsilon": _stats(rows, "epsilon"),
        },
        "attacker": {
            "gamma": _stats(rows, "attacker_gamma"),
            "local_steps": _stats(rows, "attacker_local_steps"),
            "action_norm": _stats(rows, "attacker_action_norm"),
        },
        "correlation": {
            "alpha_vs_gamma": _corr(rows, "alpha", "attacker_gamma"),
            "epsilon_vs_gamma": _corr(rows, "epsilon", "attacker_gamma"),
            "post_acc_vs_gamma": _corr(rows, "post_clean_acc", "attacker_gamma"),
            "post_acc_vs_alpha": _corr(rows, "post_clean_acc", "alpha"),
        },
    }


def _values(rows: list[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            vals.append(value)
    return vals


def _stats(rows: list[dict], key: str) -> dict[str, float]:
    vals = _values(rows, key)
    if not vals:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "final": math.nan}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
    }


def _corr(rows: list[dict], left: str, right: str) -> float:
    pairs = []
    for row in rows:
        try:
            lval = float(row.get(left, math.nan))
            rval = float(row.get(right, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(lval) and math.isfinite(rval):
            pairs.append((lval, rval))
    if len(pairs) < 2:
        return math.nan
    arr = np.asarray(pairs, dtype=np.float64)
    if float(np.std(arr[:, 0])) == 0.0 or float(np.std(arr[:, 1])) == 0.0:
        return math.nan
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_interaction(rows: list[dict], output_dir: Path) -> Path:
    steps = np.asarray(_values(rows, "step"), dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle("RL Attacker vs RL Defender Interaction Eval", fontsize=14, fontweight="bold")

    _plot_line(axes[0], steps, rows, "post_clean_acc", "Post-clean acc", "#0072BD")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(True, alpha=0.25)

    _plot_line(axes[1], steps, rows, "alpha", "defender alpha", "#0072BD")
    _plot_line(axes[1], steps, rows, "beta", "defender beta", "#D95319")
    _plot_line(axes[1], steps, rows, "epsilon", "defender epsilon", "#77AC30")
    axes[1].set_ylabel("defense action")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(True, alpha=0.25)

    _plot_line(axes[2], steps, rows, "attacker_gamma", "attacker gamma", "#A2142F")
    ax2 = axes[2].twinx()
    _plot_line(ax2, steps, rows, "attacker_local_steps", "attacker local steps", "#7E2F8E")
    axes[2].set_ylabel("gamma")
    ax2.set_ylabel("local steps")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlabel("FL eval step")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "rl_interaction_plot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_line(ax, steps: np.ndarray, rows: list[dict], key: str, label: str, color: str) -> None:
    pairs = []
    for row in rows:
        try:
            step = float(row.get("step", math.nan))
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(step) and math.isfinite(value):
            pairs.append((step, value))
    if not pairs:
        return
    arr = np.asarray(pairs, dtype=np.float64)
    steps = arr[:, 0]
    vals = arr[:, 1]
    ax.plot(steps, vals, label=label, color=color, linewidth=1.8)
    ax.scatter(steps, vals, color=color, s=8, alpha=0.35, linewidths=0)


if __name__ == "__main__":
    main()
