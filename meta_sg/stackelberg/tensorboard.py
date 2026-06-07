"""Optional TensorBoard logging helpers for Paper3D defender experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def create_summary_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover
        return None
    return SummaryWriter(str(log_dir))


def log_train_row(writer, row: dict, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("train/reward", row["defender_reward"], step)
    writer.add_scalar("train/post_clean_loss", row["post_clean_loss"], step)
    writer.add_scalar("train/post_clean_acc", row["post_clean_acc"], step)
    writer.add_scalar("train/alpha", row["alpha"], step)
    writer.add_scalar("train/beta", row["beta"], step)
    writer.add_scalar("train/epsilon", row["epsilon"], step)
    writer.add_scalar("train/mean_malicious_norm", row["mean_malicious_norm"], step)
    if np.isfinite(float(row.get("critic_loss", np.nan))):
        writer.add_scalar("td3/critic_loss", row["critic_loss"], step)
    if np.isfinite(float(row.get("actor_loss", np.nan))):
        writer.add_scalar("td3/actor_loss", row["actor_loss"], step)


def log_eval_summaries(writer, learned_summary: dict, baselines: dict[str, dict]) -> None:
    if writer is None:
        return
    writer.add_scalar("eval/learned_reward", learned_summary["mean_defender_reward"], 0)
    writer.add_scalar("eval/learned_post_clean_loss", learned_summary["mean_post_clean_loss"], 0)
    for name, summary in baselines.items():
        writer.add_scalar(f"eval_fixed/{name}_reward", summary["mean_defender_reward"], 0)
        writer.add_scalar(f"eval_fixed/{name}_post_clean_loss", summary["mean_post_clean_loss"], 0)
