"""Postprocess separate clean and attack outputs into a shared TensorBoard view."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess clean and attack outputs into TensorBoard runs")
    parser.add_argument("--clean_input_dir", type=str, required=True)
    parser.add_argument("--attack_input_dir", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--skip_clean_write", action="store_true")
    return parser.parse_args()


def _resolve_summary_path(input_dir: Path, label: str) -> Path:
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        return summary_path

    parent = input_dir.parent if input_dir.parent.exists() else None
    available = []
    if parent is not None:
        available = sorted(
            candidate.name
            for candidate in parent.iterdir()
            if candidate.is_dir() and (candidate / "summary.json").exists()
        )

    message_lines = [f"Missing {label} summary: {summary_path}"]
    if available:
        similar = difflib.get_close_matches(input_dir.name, available, n=5, cutoff=0.0)
        if similar:
            message_lines.append("Available directories with summary.json:")
            message_lines.extend(f"  - {name}" for name in similar)
    raise FileNotFoundError("\n".join(message_lines))


def _load_clean_series(clean_input_dir: Path) -> tuple[dict, dict[str, list[float]]]:
    with _resolve_summary_path(clean_input_dir, "clean").open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if "runs" in payload and "clean" in payload["runs"]:
        return payload.get("config", {}), payload["runs"]["clean"]

    rounds = payload["rounds"]
    series = {
        "clean_loss": [float(item["clean_loss"]) for item in rounds],
        "clean_acc": [float(item["clean_acc"]) for item in rounds],
        "backdoor_acc": [0.0 for _ in rounds],
        "round_seconds": [float(item["round_seconds"]) for item in rounds],
        "mean_benign_norm": [0.0 for _ in rounds],
        "mean_malicious_norm": [0.0 for _ in rounds],
        "mean_malicious_cosine": [0.0 for _ in rounds],
    }
    return payload.get("config", {}), series


def _load_attack_series(attack_input_dir: Path) -> tuple[dict, str, dict[str, list[float]]]:
    with _resolve_summary_path(attack_input_dir, "attack").open(encoding="utf-8") as fh:
        payload = json.load(fh)

    attack_type = payload.get("attack_type") or payload.get("config", {}).get("attack_type", "attack")
    if "series" in payload:
        return payload.get("config", {}), attack_type, payload["series"]
    if "runs" in payload:
        if attack_type in payload["runs"]:
            return payload.get("config", {}), attack_type, payload["runs"][attack_type]
        attack_keys = [key for key in payload["runs"] if key != "clean"]
        if len(attack_keys) == 1:
            attack_type = attack_keys[0]
            return payload.get("config", {}), attack_type, payload["runs"][attack_type]
    raise ValueError(f"Could not find attack series in {attack_input_dir / 'summary.json'}")


def _write_common_scalars(writer: SummaryWriter, series: dict[str, list[float]]) -> None:
    for round_idx, value in enumerate(series["clean_loss"], start=1):
        writer.add_scalar("metrics/loss", value, round_idx)
    for round_idx, value in enumerate(series["clean_acc"], start=1):
        writer.add_scalar("metrics/accuracy", value, round_idx)
    for round_idx, value in enumerate(series.get("backdoor_acc", []), start=1):
        writer.add_scalar("metrics/backdoor_accuracy", value, round_idx)

    elapsed_seconds = np.cumsum(series["round_seconds"]).tolist()
    for round_idx, value in enumerate(series["round_seconds"], start=1):
        writer.add_scalar("metrics/round_duration_seconds", value, round_idx)
    for round_idx, value in enumerate(elapsed_seconds, start=1):
        writer.add_scalar("metrics/elapsed_seconds", value, round_idx)
    for round_idx, value in enumerate(series["mean_benign_norm"], start=1):
        writer.add_scalar("metrics/mean_benign_norm", value, round_idx)


def _write_attack_scalars(writer: SummaryWriter, series: dict[str, list[float]]) -> None:
    for round_idx, value in enumerate(series["mean_malicious_norm"], start=1):
        writer.add_scalar("attack_only/mean_malicious_norm", value, round_idx)
    for round_idx, value in enumerate(series["mean_malicious_cosine"], start=1):
        writer.add_scalar("attack_only/mean_malicious_cosine", value, round_idx)


def _reset_tb_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _build_summary_text(run_name: str, config: dict, series: dict[str, list[float]], include_attack: bool) -> str:
    total_seconds = float(np.sum(series["round_seconds"]))
    lines = [
        f"run: {run_name}",
        f"dataset: {config.get('dataset', 'unknown')}",
        f"rounds: {len(series['clean_acc'])}",
        f"num_clients: {config.get('num_clients', 'unknown')}",
        f"subsample_rate: {config.get('subsample_rate', 'unknown')}",
        f"local_epochs: {config.get('local_epochs', 'unknown')}",
        f"lr: {config.get('lr', 'unknown')}",
        f"batch_size: {config.get('batch_size', 'unknown')}",
        f"final_accuracy: {series['clean_acc'][-1]:.4f}",
        f"final_backdoor_accuracy: {series.get('backdoor_acc', [0.0])[-1]:.4f}",
        f"final_loss: {series['clean_loss'][-1]:.4f}",
        f"mean_round_duration_seconds: {total_seconds / len(series['round_seconds']):.4f}",
        f"total_elapsed_seconds: {total_seconds:.4f}",
    ]
    if include_attack:
        lines.append(f"final_mean_malicious_norm: {series['mean_malicious_norm'][-1]:.4f}")
        lines.append(f"final_mean_malicious_cosine: {series['mean_malicious_cosine'][-1]:.4f}")
    return "\n".join(lines)


def _save_attack_pngs(attack_input_dir: Path, attack_name: str, attack_series: dict[str, list[float]]) -> list[plt.Figure]:
    label = attack_name.upper()

    cosine_fig = plot_series(
        attack_series["mean_malicious_cosine"],
        title=f"{label} Cosine to Benign Mean Update",
        ylabel="Cosine Similarity",
        save_path=str(attack_input_dir / f"{attack_name}_cosine_to_benign.png"),
    )
    cosine_ax = cosine_fig.axes[0]
    cosine_values = np.asarray(attack_series["mean_malicious_cosine"], dtype=float)
    if cosine_values.size:
        ymin = float(np.min(cosine_values))
        ymax = float(np.max(cosine_values))
        pad = max((ymax - ymin) * 0.35, 5e-7)
        cosine_ax.set_ylim(ymin - pad, ymax + pad)
        cosine_ax.axhline(-1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.8)
        cosine_ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.35)
    cosine_fig.tight_layout()
    cosine_fig.savefig(attack_input_dir / f"{attack_name}_cosine_to_benign.png", dpi=160, bbox_inches="tight")

    backdoor_fig = plot_series(
        attack_series.get("backdoor_acc", [0.0] * len(attack_series["clean_acc"])),
        title=f"{label} Backdoor Accuracy",
        ylabel="Accuracy",
        save_path=str(attack_input_dir / f"{attack_name}_backdoor_acc.png"),
    )
    backdoor_fig.tight_layout()
    backdoor_fig.savefig(attack_input_dir / f"{attack_name}_backdoor_acc.png", dpi=160, bbox_inches="tight")

    norm_fig = plot_series(
        attack_series["mean_malicious_norm"],
        title=f"{label} Malicious Update Norm",
        ylabel="L2 Norm",
        save_path=str(attack_input_dir / f"{attack_name}_malicious_norm.png"),
    )
    norm_ax = norm_fig.axes[0]
    norm_values = np.asarray(attack_series["mean_malicious_norm"], dtype=float)
    if norm_values.size:
        ymin = float(np.min(norm_values))
        ymax = float(np.max(norm_values))
        pad = max((ymax - ymin) * 0.2, 1e-6)
        norm_ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    norm_fig.tight_layout()
    norm_fig.savefig(attack_input_dir / f"{attack_name}_malicious_norm.png", dpi=160, bbox_inches="tight")

    return [cosine_fig, backdoor_fig, norm_fig]


def main() -> None:
    global plt, np, SummaryWriter, plot_series

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

    from attacker_sandbox.visualize import plot_series

    args = parse_args()
    clean_input_dir = Path(args.clean_input_dir)
    attack_input_dir = Path(args.attack_input_dir)

    clean_config, clean_series = _load_clean_series(clean_input_dir)
    attack_config, attack_type, attack_series = _load_attack_series(attack_input_dir)

    if len(clean_series["clean_acc"]) != len(attack_series["clean_acc"]):
        raise ValueError(
            f"Round mismatch between clean ({len(clean_series['clean_acc'])}) and "
            f"{attack_type} ({len(attack_series['clean_acc'])})."
        )

    figures = _save_attack_pngs(attack_input_dir, attack_type, attack_series)

    if args.tb_dir:
        tb_dir = Path(args.tb_dir)
        clean_run_dir = tb_dir / "clean"
        attack_run_dir = tb_dir / attack_type
        _reset_tb_run_dir(attack_run_dir)
        if not args.skip_clean_write:
            _reset_tb_run_dir(clean_run_dir)

        attack_writer = SummaryWriter(log_dir=str(attack_run_dir))
        _write_common_scalars(attack_writer, attack_series)
        _write_attack_scalars(attack_writer, attack_series)

        attack_writer.add_text("config/json", json.dumps(attack_config, indent=2), global_step=0)
        attack_writer.add_text(
            "run/summary",
            _build_summary_text(attack_type, attack_config, attack_series, include_attack=True),
            global_step=0,
        )
        attack_writer.flush()
        attack_writer.close()

        if not args.skip_clean_write:
            clean_writer = SummaryWriter(log_dir=str(clean_run_dir))
            _write_common_scalars(clean_writer, clean_series)
            clean_writer.add_text("config/json", json.dumps(clean_config, indent=2), global_step=0)
            clean_writer.add_text(
                "run/summary",
                _build_summary_text("clean", clean_config, clean_series, include_attack=False),
                global_step=0,
            )
            clean_writer.flush()
            clean_writer.close()

    for fig in figures:
        plt.close(fig)

    print(f"Postprocess finished for clean: {clean_input_dir}")
    print(f"Postprocess finished for attack: {attack_input_dir}")
    if args.tb_dir:
        print(f"TensorBoard run written to: {Path(args.tb_dir) / 'clean'}")
        print(f"TensorBoard run written to: {Path(args.tb_dir) / attack_type}")


if __name__ == "__main__":
    main()
