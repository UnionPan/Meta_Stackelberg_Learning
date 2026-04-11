"""Postprocess sandbox comparison outputs into figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_sandbox.visualize import plot_dual_series, plot_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess sandbox comparison outputs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"

    with summary_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    clean_series = payload["runs"]["clean"]
    ipm_series = payload["runs"]["ipm"]

    figures = {
        "clean_loss_compare": plot_dual_series(
            clean_series["clean_loss"],
            ipm_series["clean_loss"],
            label_a="clean FL",
            label_b="IPM attack",
            title="Clean Loss Across Rounds",
            ylabel="Cross Entropy Loss",
            save_path=str(input_dir / "clean_loss_compare.png"),
        ),
        "clean_acc_compare": plot_dual_series(
            clean_series["clean_acc"],
            ipm_series["clean_acc"],
            label_a="clean FL",
            label_b="IPM attack",
            title="Clean Accuracy Across Rounds",
            ylabel="Accuracy",
            save_path=str(input_dir / "clean_acc_compare.png"),
        ),
        "malicious_norm_compare": plot_dual_series(
            clean_series["mean_malicious_norm"],
            ipm_series["mean_malicious_norm"],
            label_a="clean FL",
            label_b="IPM attack",
            title="Mean Malicious Update Norm",
            ylabel="L2 Norm",
            save_path=str(input_dir / "malicious_norm_compare.png"),
        ),
        "ipm_cosine_to_benign": plot_series(
            ipm_series["mean_malicious_cosine"],
            title="IPM Cosine to Benign Mean Update",
            ylabel="Cosine Similarity",
            save_path=str(input_dir / "ipm_cosine_to_benign.png"),
        ),
    }

    if args.tb_dir:
        writer = SummaryWriter(log_dir=args.tb_dir)
        for round_idx, value in enumerate(clean_series["clean_loss"], start=1):
            writer.add_scalar("clean/loss", value, round_idx)
        for round_idx, value in enumerate(clean_series["clean_acc"], start=1):
            writer.add_scalar("clean/accuracy", value, round_idx)
        for round_idx, value in enumerate(ipm_series["clean_loss"], start=1):
            writer.add_scalar("ipm/loss", value, round_idx)
        for round_idx, value in enumerate(ipm_series["clean_acc"], start=1):
            writer.add_scalar("ipm/accuracy", value, round_idx)
        for round_idx, value in enumerate(ipm_series["mean_malicious_norm"], start=1):
            writer.add_scalar("ipm/mean_malicious_norm", value, round_idx)
        for round_idx, value in enumerate(ipm_series["mean_malicious_cosine"], start=1):
            writer.add_scalar("ipm/mean_malicious_cosine", value, round_idx)
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
