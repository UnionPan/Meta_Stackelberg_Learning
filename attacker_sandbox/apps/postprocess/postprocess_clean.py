"""Postprocess clean benchmark outputs into figures and TensorBoard images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess clean benchmark outputs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    global plt, np, SummaryWriter
    global plot_client_metric_heatmap, plot_mean_with_band, plot_round_boxplot, plot_series, plot_xy_series

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "required package"
        raise SystemExit(
            "Missing Python dependency for postprocess: "
            f"{missing_name}. Activate the sandbox venv or run "
            "'bash attacker_sandbox/scripts/setup_env.sh cpu' first."
        ) from exc

    from attacker_sandbox.visualize import (
        plot_client_metric_heatmap,
        plot_mean_with_band,
        plot_round_boxplot,
        plot_series,
        plot_xy_series,
    )

    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"
    client_metrics_path = input_dir / "client_metrics.csv"

    with summary_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    with client_metrics_path.open(encoding="utf-8", newline="") as fh:
        client_rows = list(csv.DictReader(fh))

    rounds = payload["rounds"]
    clean_loss = [float(item["clean_loss"]) for item in rounds]
    clean_acc = [float(item["clean_acc"]) for item in rounds]
    round_seconds = [float(item["round_seconds"]) for item in rounds]
    cumulative_time = np.cumsum(round_seconds).tolist()
    round_count = len(rounds)

    per_round_client_loss = []
    per_round_client_acc = []
    per_round_update_norm = []
    selected_client_counts = []
    for round_idx in range(1, round_count + 1):
        round_rows = [row for row in client_rows if int(row["round_idx"]) == round_idx]
        selected_rows = [row for row in round_rows if row["selected"] == "True"]
        selected_client_counts.append(len(selected_rows))
        per_round_client_loss.append([float(row["train_loss"]) for row in selected_rows if row["train_loss"] not in ("", None)])
        per_round_client_acc.append([float(row["train_acc"]) for row in selected_rows if row["train_acc"] not in ("", None)])
        per_round_update_norm.append([float(row["update_norm"]) for row in selected_rows if row["update_norm"] not in ("", None)])

    mean_client_loss = [float(np.mean(values)) if values else 0.0 for values in per_round_client_loss]
    std_client_loss = [float(np.std(values)) if values else 0.0 for values in per_round_client_loss]
    mean_client_acc = [float(np.mean(values)) if values else 0.0 for values in per_round_client_acc]
    std_client_acc = [float(np.std(values)) if values else 0.0 for values in per_round_client_acc]
    mean_update_norm = [float(np.mean(values)) if values else 0.0 for values in per_round_update_norm]
    std_update_norm = [float(np.std(values)) if values else 0.0 for values in per_round_update_norm]

    figures = {
        "figures/clean_loss": plot_series(clean_loss, "Clean FL Loss", "Cross Entropy Loss", str(input_dir / "clean_loss.png")),
        "figures/clean_accuracy": plot_series(clean_acc, "Clean FL Accuracy", "Accuracy", str(input_dir / "clean_acc.png")),
        "figures/round_seconds": plot_series(round_seconds, "Clean FL Round Time", "Seconds", str(input_dir / "round_seconds.png")),
        "figures/cumulative_seconds": plot_series(cumulative_time, "Cumulative Training Time", "Seconds", str(input_dir / "cumulative_seconds.png")),
        "figures/clean_acc_vs_time": plot_xy_series(cumulative_time, clean_acc, "Clean Accuracy vs Wall-Clock Time", "Cumulative Seconds", "Accuracy", str(input_dir / "clean_acc_vs_time.png")),
        "figures/clean_loss_vs_time": plot_xy_series(cumulative_time, clean_loss, "Clean Loss vs Wall-Clock Time", "Cumulative Seconds", "Cross Entropy Loss", str(input_dir / "clean_loss_vs_time.png")),
        "figures/selected_clients": plot_series(selected_client_counts, "Selected Clients per Round", "Client Count", str(input_dir / "selected_clients.png")),
        "figures/client_train_loss_mean_std": plot_mean_with_band(mean_client_loss, std_client_loss, "Client Train Loss Mean +/- Std", "Train Loss", str(input_dir / "client_train_loss_mean_std.png")),
        "figures/client_train_acc_mean_std": plot_mean_with_band(mean_client_acc, std_client_acc, "Client Train Accuracy Mean +/- Std", "Train Accuracy", str(input_dir / "client_train_acc_mean_std.png")),
        "figures/client_update_norm_mean_std": plot_mean_with_band(mean_update_norm, std_update_norm, "Client Update Norm Mean +/- Std", "Update Norm", str(input_dir / "client_update_norm_mean_std.png")),
        "figures/client_update_norm_boxplot": plot_round_boxplot(per_round_update_norm, "Client Update Norm Distribution by Round", "Update Norm", str(input_dir / "client_update_norm_boxplot.png")),
        "figures/client_train_loss_heatmap": plot_client_metric_heatmap(client_rows, "train_loss", "Client Train Loss Heatmap", "Train Loss", str(input_dir / "client_train_loss_heatmap.png")),
        "figures/client_train_acc_heatmap": plot_client_metric_heatmap(client_rows, "train_acc", "Client Train Accuracy Heatmap", "Train Accuracy", str(input_dir / "client_train_acc_heatmap.png")),
        "figures/client_update_norm_heatmap": plot_client_metric_heatmap(client_rows, "update_norm", "Client Update Norm Heatmap", "Update Norm", str(input_dir / "client_update_norm_heatmap.png")),
    }

    if args.tb_dir:
        writer = SummaryWriter(log_dir=args.tb_dir)
        for tag, fig in figures.items():
            writer.add_figure(tag, fig, global_step=0)
        writer.flush()
        writer.close()

    for fig in figures.values():
        plt.close(fig)

    print(f"Postprocess finished for: {input_dir}")
    if args.tb_dir:
        print(f"TensorBoard figures written to: {args.tb_dir}")


if __name__ == "__main__":
    main()
