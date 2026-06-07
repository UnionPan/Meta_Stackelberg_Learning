"""Generate MATLAB-style figures for the Paper3D RL-defender experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATLAB_COLORS = {
    "blue": "#0072BD",
    "orange": "#D95319",
    "yellow": "#EDB120",
    "purple": "#7E2F8E",
    "green": "#77AC30",
    "cyan": "#4DBEEE",
    "red": "#A2142F",
}

SCENARIOS = [
    ("Clean", "clean_no_attack", "clean_no_attack", MATLAB_COLORS["green"], "o"),
    ("No defense", "frozen_attacker_no_defense", "attacker_no_defense", MATLAB_COLORS["red"], "s"),
    ("Fixed trim", "frozen_attacker_fixed_trim", "fixed_trim", MATLAB_COLORS["orange"], "^"),
    ("RL defender", "frozen_attacker_learned_defender", "learned_defender", MATLAB_COLORS["blue"], "d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-dir",
        default=(
            "runs/stackelberg_paper3d_formal_eval/"
            "formal_paper3d_formal_eval_newdefender1000_warmup100_seed7_20260606_224050"
        ),
    )
    parser.add_argument(
        "--train-dir",
        default=(
            "runs/stackelberg_paper3d_defender_td3/"
            "fl_sandbox_paper3d_full_neuroclip_sharedwarmup100_td3_1000s_seed7_20260606_220922"
        ),
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def configure_matlab_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data(formal_dir: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    summary = json.loads((formal_dir / "summary.json").read_text())
    frames = {
        label: pd.read_csv(formal_dir / csv_dir / "rounds.csv")
        for label, _, csv_dir, _, _ in SCENARIOS
    }
    return summary, frames


def save_all(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{name}.{ext}")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_main_result(summary: dict, frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    labels = [item[0] for item in SCENARIOS]
    keys = [item[1] for item in SCENARIOS]
    colors = [item[3] for item in SCENARIOS]
    acc = np.array([summary[key]["mean_post_clean_acc"] for key in keys])
    loss = np.array([summary[key]["mean_post_clean_loss"] for key in keys])
    final_acc = np.array([summary[key]["final_post_clean_acc"] for key in keys])

    clean_acc = summary["clean_no_attack"]["mean_post_clean_acc"]
    no_def_acc = summary["frozen_attacker_no_defense"]["mean_post_clean_acc"]
    recovery = (acc - no_def_acc) / max(clean_acc - no_def_acc, 1e-12)
    recovery[0] = 1.0
    recovery[1] = 0.0

    fig = plt.figure(figsize=(13.2, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], wspace=0.34, hspace=0.42)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_rec = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, 0:2])
    ax_action = fig.add_subplot(gs[1, 2])

    x = np.arange(len(labels))
    ax_acc.bar(x, acc, color=colors, width=0.68, edgecolor="black", linewidth=0.6)
    ax_acc.plot(x, final_acc, "k--", marker="x", linewidth=1.3, label="final")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, rotation=22, ha="right")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_title("Mean defended accuracy")
    for i, value in enumerate(acc):
        ax_acc.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    ax_acc.legend(frameon=True, loc="lower right")
    panel_label(ax_acc, "A")

    ax_loss.bar(x, loss, color=colors, width=0.68, edgecolor="black", linewidth=0.6)
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(labels, rotation=22, ha="right")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Mean defended loss")
    for i, value in enumerate(loss):
        ax_loss.text(i, value + max(loss) * 0.025, f"{value:.2f}", ha="center", fontsize=9)
    panel_label(ax_loss, "B")

    ax_rec.bar(x, recovery, color=colors, width=0.68, edgecolor="black", linewidth=0.6)
    ax_rec.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels(labels, rotation=22, ha="right")
    ax_rec.set_ylabel("Recovery ratio")
    ax_rec.set_ylim(-0.08, max(1.08, float(recovery.max()) + 0.15))
    ax_rec.set_title("Accuracy recovery")
    for i, value in enumerate(recovery):
        ax_rec.text(i, value + 0.035, f"{value:.2f}", ha="center", fontsize=9)
    panel_label(ax_rec, "C")

    for label, _, _, color, marker in SCENARIOS:
        df = frames[label]
        ax_curve.plot(
            df["step"],
            df["post_clean_acc"],
            color=color,
            marker=marker,
            markevery=max(1, len(df) // 5),
            label=label,
        )
    ax_curve.set_xlabel("FL evaluation step")
    ax_curve.set_ylabel("Post-clean accuracy")
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_title("Accuracy trajectory under frozen attacker")
    ax_curve.legend(ncol=2, frameon=True, loc="lower left")
    panel_label(ax_curve, "D")

    learned = frames["RL defender"]
    fixed = frames["Fixed trim"]
    action_names = ["alpha", "beta", "epsilon"]
    idx = np.arange(len(action_names))
    width = 0.34
    ax_action.bar(
        idx - width / 2,
        [fixed[name].mean() for name in action_names],
        width=width,
        color=MATLAB_COLORS["orange"],
        edgecolor="black",
        linewidth=0.6,
        label="Fixed trim",
    )
    ax_action.bar(
        idx + width / 2,
        [learned[name].mean() for name in action_names],
        width=width,
        color=MATLAB_COLORS["blue"],
        edgecolor="black",
        linewidth=0.6,
        label="RL defender",
    )
    ax_action.set_xticks(idx)
    ax_action.set_xticklabels([r"$\alpha$", r"$\beta$", r"$\epsilon$"])
    ax_action.set_ylabel("Action value")
    ax_action.set_title("Defender action")
    ax_action.legend(frameon=True)
    panel_label(ax_action, "E")

    fig.suptitle("RL Defender Against Frozen RL Attacker", fontsize=15, fontweight="bold", y=0.99)
    save_all(fig, output_dir, "paper3d_matlab_main_result")
    plt.close(fig)


def plot_training_and_diagnostics(train_dir: Path, frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    train = pd.read_csv(train_dir / "rounds.csv")
    roll = 25
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    axes = axes.ravel()

    axes[0].plot(train["step"], train["post_clean_acc"], color=MATLAB_COLORS["blue"], alpha=0.22)
    axes[0].plot(
        train["step"],
        train["post_clean_acc"].rolling(roll, min_periods=1).mean(),
        color=MATLAB_COLORS["blue"],
        linewidth=2.4,
        label=f"{roll}-step mean",
    )
    axes[0].set_title("Training defended accuracy")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Post-clean accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=True)
    panel_label(axes[0], "A")

    axes[1].plot(train["step"], train["post_clean_loss"], color=MATLAB_COLORS["red"], alpha=0.22)
    axes[1].plot(
        train["step"],
        train["post_clean_loss"].rolling(roll, min_periods=1).mean(),
        color=MATLAB_COLORS["red"],
        linewidth=2.4,
    )
    axes[1].set_title("Training defended loss")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Post-clean loss")
    axes[1].set_yscale("symlog", linthresh=2)
    panel_label(axes[1], "B")

    for name, color in [
        ("alpha", MATLAB_COLORS["blue"]),
        ("beta", MATLAB_COLORS["orange"]),
        ("epsilon", MATLAB_COLORS["green"]),
    ]:
        axes[2].plot(
            train["step"],
            train[name].rolling(roll, min_periods=1).mean(),
            color=color,
            linewidth=2.1,
            label=name,
        )
    axes[2].set_title("Learned action convergence")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Action value")
    axes[2].legend(frameon=True)
    panel_label(axes[2], "C")

    learned = frames["RL defender"]
    no_def = frames["No defense"]
    fixed = frames["Fixed trim"]
    axes[3].plot(no_def["step"], no_def["mean_malicious_norm"], color=MATLAB_COLORS["red"], marker="s", label="No defense")
    axes[3].plot(fixed["step"], fixed["mean_malicious_norm"], color=MATLAB_COLORS["orange"], marker="^", label="Fixed trim")
    axes[3].plot(learned["step"], learned["mean_malicious_norm"], color=MATLAB_COLORS["blue"], marker="d", label="RL defender")
    axes[3].set_title("Observed malicious update norm")
    axes[3].set_xlabel("FL evaluation step")
    axes[3].set_ylabel("Mean malicious norm")
    axes[3].legend(frameon=True)
    panel_label(axes[3], "D")

    fig.suptitle("Training Dynamics and Attack Diagnostics", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_all(fig, output_dir, "paper3d_matlab_training_diagnostics")
    plt.close(fig)


def plot_metric_improvement(summary: dict, output_dir: Path) -> None:
    no_def = summary["frozen_attacker_no_defense"]
    fixed = summary["frozen_attacker_fixed_trim"]
    learned = summary["frozen_attacker_learned_defender"]
    clean = summary["clean_no_attack"]

    metrics = {
        "Attack drop\n(clean - no defense)": clean["mean_post_clean_acc"] - no_def["mean_post_clean_acc"],
        "Fixed gain\n(fixed - no defense)": fixed["mean_post_clean_acc"] - no_def["mean_post_clean_acc"],
        "RL gain\n(RL - no defense)": learned["mean_post_clean_acc"] - no_def["mean_post_clean_acc"],
        "RL over fixed\n(RL - fixed)": learned["mean_post_clean_acc"] - fixed["mean_post_clean_acc"],
    }
    colors = [MATLAB_COLORS["red"], MATLAB_COLORS["orange"], MATLAB_COLORS["blue"], MATLAB_COLORS["purple"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(metrics))
    values = list(metrics.values())
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylabel("Accuracy delta")
    ax.set_ylim(min(0, min(values) - 0.04), max(values) + 0.06)
    ax.set_title("Defense effect size", pad=12)
    for i, value in enumerate(values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.01 if value >= 0 else -0.01
        ax.text(i, value + offset, f"{value:+.3f}", ha="center", va=va)
    fig.tight_layout()
    save_all(fig, output_dir, "paper3d_matlab_effect_size")
    plt.close(fig)


def write_result_table(summary: dict, output_dir: Path) -> None:
    rows = []
    for label, key, _, _, _ in SCENARIOS:
        item = summary[key]
        rows.append(
            {
                "scenario": label,
                "mean_acc": item["mean_post_clean_acc"],
                "mean_loss": item["mean_post_clean_loss"],
                "final_acc": item["final_post_clean_acc"],
                "mean_alpha": item["mean_alpha"],
                "mean_beta": item["mean_beta"],
                "mean_epsilon": item["mean_epsilon"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "paper3d_result_table.csv", index=False)
    latex = table.to_latex(index=False, float_format="%.4f")
    (output_dir / "paper3d_result_table.tex").write_text(latex)


def main() -> None:
    args = parse_args()
    configure_matlab_style()
    formal_dir = Path(args.formal_dir)
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir) if args.output_dir else formal_dir / "plots_matlab"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, frames = load_data(formal_dir)
    plot_main_result(summary, frames, output_dir)
    plot_training_and_diagnostics(train_dir, frames, output_dir)
    plot_metric_improvement(summary, output_dir)
    write_result_table(summary, output_dir)

    print(output_dir)
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
