"""Metric rows, summaries, and comparisons for Paper3D defender experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from meta_sg.stackelberg.paper3d_action import Paper3DAction


def make_round_row(
    *,
    global_step: int,
    episode: int,
    info: dict,
    raw_action: np.ndarray,
    action: Paper3DAction,
    defender_reward: float,
    attacker_reward: float,
    episode_return: float,
    stats: dict,
) -> dict:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw.shape[0] < 3:
        raw = np.pad(raw, (0, 3 - raw.shape[0]))
    malicious_norms = np.asarray(info.get("malicious_update_norms", []), dtype=np.float32)
    attack_metrics = dict(info.get("attack_metrics", {}) or {})
    return {
        "step": int(global_step),
        "episode": int(episode),
        "round": int(info["round"]),
        "raw_alpha_action": float(raw[0]),
        "raw_beta_action": float(raw[1]),
        "raw_epsilon_action": float(raw[2]),
        "alpha": float(action.alpha),
        "beta": float(action.beta),
        "epsilon": float(action.epsilon),
        "defender_reward": float(defender_reward),
        "attacker_reward": float(attacker_reward),
        "episode_defender_return": float(episode_return),
        "clean_acc": float(info["clean_acc"]),
        "clean_loss": float(info["clean_loss"]),
        "post_clean_acc": float(info.get("post_clean_acc", info["clean_acc"])),
        "post_clean_loss": float(info.get("post_clean_loss", info["clean_loss"])),
        "backdoor_acc": float(info["backdoor_acc"]),
        "post_backdoor_acc": float(info.get("post_backdoor_acc", info["backdoor_acc"])),
        "mean_malicious_norm": float(np.mean(malicious_norms)) if malicious_norms.size else 0.0,
        "attacker_gamma": float(attack_metrics.get("rl_action_gamma", np.nan)),
        "attacker_local_steps": float(attack_metrics.get("rl_action_local_steps", np.nan)),
        "attacker_action_norm": float(attack_metrics.get("rl_action_norm", np.nan)),
        "critic_loss": float(stats.get("critic_loss", np.nan)),
        "actor_loss": float(stats.get("actor_loss", np.nan)),
    }


def summarize_rows(rows: list[dict], output_dir: Path) -> dict:
    rewards = [float(row["defender_reward"]) for row in rows] or [0.0]
    post_losses = [float(row["post_clean_loss"]) for row in rows] or [0.0]
    post_accs = [float(row["post_clean_acc"]) for row in rows] or [0.0]
    losses = [float(row.get("clean_loss", 0.0)) for row in rows] or [0.0]
    accs = [float(row.get("clean_acc", 0.0)) for row in rows] or [0.0]
    alphas = [float(row["alpha"]) for row in rows] or [0.0]
    betas = [float(row["beta"]) for row in rows] or [0.0]
    epsilons = [float(row["epsilon"]) for row in rows] or [0.0]
    return {
        "steps": len(rows),
        "mean_defender_reward": float(np.mean(rewards)),
        "mean_clean_loss": float(np.mean(losses)),
        "mean_clean_acc": float(np.mean(accs)),
        "mean_post_clean_loss": float(np.mean(post_losses)),
        "final_post_clean_acc": float(post_accs[-1]),
        "mean_post_clean_acc": float(np.mean(post_accs)),
        "mean_alpha": float(np.mean(alphas)),
        "mean_beta": float(np.mean(betas)),
        "mean_epsilon": float(np.mean(epsilons)),
        "rounds_csv": str(output_dir / "rounds.csv"),
    }


def compare_learned_to_fixed_paper3d(learned: dict, baselines: dict[str, dict]) -> dict:
    if not baselines:
        return {}
    best_name, best = max(
        baselines.items(),
        key=lambda item: float(item[1].get("mean_defender_reward", float("-inf"))),
    )
    return {
        "best_fixed_baseline": best_name,
        "defender_reward_improvement_vs_best_fixed": float(learned["mean_defender_reward"])
        - float(best["mean_defender_reward"]),
        "post_clean_loss_delta_vs_best_fixed": float(learned["mean_post_clean_loss"])
        - float(best["mean_post_clean_loss"]),
    }


def format_progress_line(phase: str, row: dict) -> str:
    return (
        f"[{phase}] "
        f"step={int(row['step'])} "
        f"round={int(row['round'])} "
        f"reward={float(row['defender_reward']):.6f} "
        f"post_loss={float(row['post_clean_loss']):.6f} "
        f"post_acc={float(row['post_clean_acc']):.6f} "
        f"alpha={float(row['alpha']):.4f} "
        f"beta={float(row['beta']):.4f} "
        f"epsilon={float(row['epsilon']):.4f} "
        f"mal_norm={float(row['mean_malicious_norm']):.4f}"
    )


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
