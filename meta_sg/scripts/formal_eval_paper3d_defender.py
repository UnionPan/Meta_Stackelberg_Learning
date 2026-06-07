"""Formal Paper3D defender evaluation with clean and attacked baselines."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch

from fl_sandbox.aggregators.rules import fedavg_aggregate

from meta_sg.learning.config import TD3Config
from meta_sg.learning.td3 import TD3Agent
from meta_sg.scripts.run_stackelberg_defender_td3 import (
    build_env,
    resolve_attacker_checkpoint,
    resolve_distribution_dir,
)
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.stackelberg.metrics import format_progress_line, make_round_row, summarize_rows, write_rows
from meta_sg.stackelberg.paper3d_action import Paper3DAction, fixed_paper3d_to_raw_action, raw_action_to_paper3d
from meta_sg.stackelberg.paper3d_sandbox import FrozenSandboxAttack
from meta_sg.stackelberg.paper_reward import compute_paper_defender_reward
from meta_sg.stackelberg.warmup import clone_weights, reset_env_from_weights
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy


@dataclass
class FedAvgAsPaperDefender:
    defense_type: str = "paper_norm_trimmed_mean"

    def aggregate(self, old_weights, new_weights, *, trusted_weights=None):
        del old_weights, trusted_weights
        return fedavg_aggregate(new_weights)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defender-checkpoint", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="runs/stackelberg_paper3d_formal_eval")
    parser.add_argument("--attacker-checkpoint", default="")
    parser.add_argument("--rl-distribution-dir", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-horizon", type=int, default=5)
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
    parser.add_argument("--fixed-trim", default="fixed_trim=4,0.2,5")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    attacker_checkpoint = resolve_attacker_checkpoint(args.attacker_checkpoint)
    distribution_dir = resolve_distribution_dir(args.rl_distribution_dir, attacker_checkpoint)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / f"formal_paper3d_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_weights, warmup_metrics = run_clean_warmup(args, output_dir=output_dir)
    clean_rows = evaluate_clean_no_attack(args, output_dir=output_dir, warmup_weights=warmup_weights)
    attacked_no_defense_rows = evaluate_attacker_no_defense(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        output_dir=output_dir,
        warmup_weights=warmup_weights,
    )
    fixed_name, fixed_action = parse_one_fixed_trim(args.fixed_trim)
    fixed_rows = evaluate_fixed_from_warmup(
        args,
        fixed_action,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        warmup_weights=warmup_weights,
    )
    fixed_dir = output_dir / fixed_name
    write_rows(fixed_dir / "rounds.csv", fixed_rows)

    learned = load_defender(args, attacker_checkpoint=attacker_checkpoint, distribution_dir=distribution_dir)
    learned_rows = evaluate_learned_from_warmup(
        args,
        learned,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        warmup_weights=warmup_weights,
    )
    learned_dir = output_dir / "learned_defender"
    write_rows(learned_dir / "rounds.csv", learned_rows)

    summary = {
        "defender_checkpoint": str(Path(args.defender_checkpoint).expanduser()),
        "attacker_checkpoint": str(attacker_checkpoint),
        "rl_distribution_dir": str(distribution_dir),
        "eval_samples": int(args.eval_samples),
        "warmup_rounds": int(args.warmup_rounds),
        "warmup_metrics": warmup_metrics,
        "horizon": int(args.eval_horizon or args.horizon),
        "eval_episodes": int(args.eval_episodes),
        "clean_no_attack": summarize_rows(clean_rows, output_dir / "clean_no_attack"),
        "frozen_attacker_no_defense": summarize_rows(attacked_no_defense_rows, output_dir / "attacker_no_defense"),
        "frozen_attacker_fixed_trim": summarize_rows(fixed_rows, fixed_dir),
        "frozen_attacker_learned_defender": summarize_rows(learned_rows, learned_dir),
    }
    summary["comparison"] = compare_formal(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "FORMAL_PAPER3D_EVAL_RESULT",
        f"clean_acc={summary['clean_no_attack']['mean_post_clean_acc']:.6f}",
        f"no_defense_acc={summary['frozen_attacker_no_defense']['mean_post_clean_acc']:.6f}",
        f"fixed_trim_acc={summary['frozen_attacker_fixed_trim']['mean_post_clean_acc']:.6f}",
        f"learned_acc={summary['frozen_attacker_learned_defender']['mean_post_clean_acc']:.6f}",
        f"output={output_dir}",
    )


def run_clean_warmup(args: argparse.Namespace, *, output_dir: Path) -> tuple[list[np.ndarray], dict[str, float]]:
    env = build_env_like(args, attack_type="clean", defense_type="fedavg", seed=int(args.seed) + 1_000)
    env.coordinator.reset(seed=int(args.seed) + 1_000)
    warmup_rounds = int(args.warmup_rounds)
    metrics: dict[str, float] = {}
    for round_idx in range(1, warmup_rounds + 1):
        summary = env.coordinator.runner.run_round(round_idx, attack=None, evaluate=True)
        metrics = {
            "clean_loss": float(summary.clean_loss),
            "clean_acc": float(summary.clean_acc),
            "backdoor_acc": float(summary.backdoor_acc),
        }
        if int(args.print_every) > 0 and (round_idx == 1 or round_idx % int(args.print_every) == 0):
            print(
                f"[warmup_clean] round={round_idx} loss={metrics['clean_loss']:.6f} "
                f"acc={metrics['clean_acc']:.6f}",
                flush=True,
            )
    weights = clone_weights(env.coordinator.current_weights)
    torch.save({"weights": weights, "metrics": metrics}, output_dir / "warmup_checkpoint.pt")
    return weights, metrics


def evaluate_clean_no_attack(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    env = build_env_like(args, attack_type="clean", defense_type="fedavg", seed=int(args.seed) + 20_000)
    rows = evaluate_coordinator_rows(
        args,
        env,
        raw_action=np.zeros(3, dtype=np.float32),
        phase="clean_no_attack",
        warmup_weights=warmup_weights,
    )
    write_rows(output_dir / "clean_no_attack" / "rounds.csv", rows)
    return rows


def evaluate_attacker_no_defense(
    args: argparse.Namespace,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    output_dir: Path,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    env = build_plain_env_like(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        attack_type="rl",
        defense_type="paper_norm_trimmed_mean",
        seed=int(args.seed) + 30_000,
    )
    env.coordinator.runner.defender = FedAvgAsPaperDefender()
    rows = evaluate_coordinator_rows(
        args,
        env,
        raw_action=np.zeros(3, dtype=np.float32),
        phase="attacker_no_defense",
        warmup_weights=warmup_weights,
    )
    write_rows(output_dir / "attacker_no_defense" / "rounds.csv", rows)
    return rows


def build_env_like(args: argparse.Namespace, *, attack_type: str, defense_type: str, seed: int):
    return build_plain_env_like(
        args,
        attacker_checkpoint=None,
        distribution_dir=None,
        attack_type=attack_type,
        defense_type=defense_type,
        seed=seed,
    )


def build_plain_env_like(
    args: argparse.Namespace,
    *,
    attacker_checkpoint: Path | None,
    distribution_dir: Path | None,
    attack_type: str,
    defense_type: str,
    seed: int,
):
    values = dict(
        dataset="mnist",
        device=str(args.device),
        attack_type=attack_type,
        defense_type=defense_type,
        rounds=int(args.eval_horizon or args.horizon),
        start_round_idx=101,
        warmup_rounds=100,
        num_clients=int(args.num_clients),
        num_attackers=int(args.num_attackers),
        subsample_rate=float(args.subsample_rate),
        local_epochs=1,
        batch_size=int(args.batch_size_fl),
        eval_batch_size=int(args.eval_batch_size),
        fltrust_root_size=0,
        max_client_samples_per_client=int(args.client_samples),
        max_eval_samples=int(args.eval_samples),
        clipped_median_norm=4.0,
        trimmed_mean_ratio=0.2,
        seed=int(seed),
    )
    if attack_type == "rl":
        values.update(
            rl_policy_checkpoint_path=str(attacker_checkpoint),
            rl_distribution_dir=str(distribution_dir),
            rl_distribution_split="train",
            rl_distribution_steps=10,
            rl_freeze_policy=True,
            rl_policy_train_steps_per_round=0,
            rl_policy_warmup_steps=0,
            rl_policy_warmup_random_steps=0,
            rl_attack_start_round=101,
            rl_policy_train_end_round=100,
        )
    config = SandboxConfig(
        **values
    )
    return SimpleNamespace(
        coordinator=FLSandboxCoordinatorAdapter(config),
        attack_strategy=None if attack_type == "clean" else FrozenSandboxAttack(),
        raw_attack_action=np.zeros(3, dtype=np.float32),
    )


def reset_eval_episode_from_warmup(env, *, seed: int, warmup_weights: list[np.ndarray]):
    return reset_env_from_weights(env, seed=seed, weights=warmup_weights)


def evaluate_coordinator_rows(
    args: argparse.Namespace,
    env,
    *,
    raw_action: np.ndarray,
    phase: str,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    global_step = 0
    horizon = int(args.eval_horizon or args.horizon)
    for episode in range(int(args.eval_episodes)):
        reset_eval_episode_from_warmup(env, seed=int(args.seed) + 20_000 + episode, warmup_weights=warmup_weights)
        for local_step in range(horizon):
            summary = env.coordinator.run_round(
                attack=getattr(env, "attack_strategy", None),
                defense=PaperDefenseStrategy(),
                attack_decision=None,
                defense_decision=None,
                evaluate=True,
            )
            global_step += 1
            row = {
                "step": global_step,
                "episode": episode,
                "round": local_step + 1,
                "raw_alpha_action": float(raw_action[0]),
                "raw_beta_action": float(raw_action[1]),
                "raw_epsilon_action": float(raw_action[2]),
                "alpha": 0.0,
                "beta": 0.0,
                "epsilon": 0.0,
                "defender_reward": -float(summary.clean_loss) * float(args.reward_scale),
                "attacker_reward": 0.0,
                "episode_defender_return": 0.0,
                "clean_acc": float(summary.clean_acc),
                "clean_loss": float(summary.clean_loss),
                "post_clean_acc": float(summary.clean_acc),
                "post_clean_loss": float(summary.clean_loss),
                "backdoor_acc": float(summary.backdoor_acc),
                "post_backdoor_acc": float(summary.backdoor_acc),
                "mean_malicious_norm": float(np.mean(summary.malicious_update_norms))
                if summary.malicious_update_norms
                else 0.0,
                "critic_loss": float("nan"),
                "actor_loss": float("nan"),
            }
            rows.append(row)
            if int(args.print_every) > 0 and (global_step == 1 or global_step % int(args.print_every) == 0):
                print(
                    f"[{phase}] step={global_step} loss={row['post_clean_loss']:.6f} "
                    f"acc={row['post_clean_acc']:.6f} mal_norm={row['mean_malicious_norm']:.4f}",
                    flush=True,
                )
    return rows


def evaluate_fixed_from_warmup(
    args: argparse.Namespace,
    action: Paper3DAction,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed) + 10_000,
        horizon=int(args.eval_horizon or args.horizon),
    )
    raw_action = fixed_paper3d_to_raw_action(action, args)
    return evaluate_bsmg_rows_from_warmup(
        args,
        env,
        raw_action_provider=lambda obs: raw_action,
        action_provider=lambda raw: action,
        phase="eval_fixed",
        warmup_weights=warmup_weights,
    )


def evaluate_learned_from_warmup(
    args: argparse.Namespace,
    defender,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed) + 10_000,
        horizon=int(args.eval_horizon or args.horizon),
    )
    return evaluate_bsmg_rows_from_warmup(
        args,
        env,
        raw_action_provider=lambda obs: defender.get_action(obs, noise=0.0),
        action_provider=lambda raw: raw_action_to_paper3d(raw, args),
        phase="eval_learned",
        warmup_weights=warmup_weights,
    )


def evaluate_bsmg_rows_from_warmup(
    args: argparse.Namespace,
    env,
    *,
    raw_action_provider,
    action_provider,
    phase: str,
    warmup_weights: list[np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    global_step = 0
    horizon = int(args.eval_horizon or args.horizon)
    for episode in range(int(args.eval_episodes)):
        obs = reset_eval_episode_from_warmup(
            env,
            seed=int(args.seed) + 10_000 + episode,
            warmup_weights=warmup_weights,
        )
        episode_return = 0.0
        for _ in range(horizon):
            raw_action = raw_action_provider(obs)
            next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
            defender_reward = compute_paper_defender_reward(info, scale=float(args.reward_scale))
            global_step += 1
            episode_return += float(defender_reward)
            row = make_round_row(
                global_step=global_step,
                episode=episode,
                info=info,
                raw_action=raw_action,
                action=action_provider(raw_action),
                defender_reward=defender_reward,
                attacker_reward=attacker_reward,
                episode_return=episode_return,
                stats={},
            )
            rows.append(row)
            print_progress(args, phase, row)
            obs = next_obs
            if done:
                break
    return rows


def print_progress(args: argparse.Namespace, phase: str, row: dict) -> None:
    interval = max(0, int(getattr(args, "print_every", 0) or 0))
    if interval <= 0:
        return
    step = int(row["step"])
    if step == 1 or step % interval == 0:
        print(format_progress_line(phase, row), flush=True)


def load_defender(args, *, attacker_checkpoint: Path, distribution_dir: Path) -> TD3Agent:
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed) + 40_000,
        horizon=int(args.eval_horizon or args.horizon),
    )
    obs = env.reset(seed=int(args.seed) + 40_000)
    cfg = TD3Config(hidden_dim=int(args.hidden_dim), batch_size=int(args.batch_size), buffer_capacity=int(args.buffer_capacity))
    defender = TD3Agent(obs_dim=int(obs.shape[0]), act_dim=3, config=cfg, device=torch.device(args.device))
    defender.load(str(Path(args.defender_checkpoint).expanduser()))
    return defender


def parse_one_fixed_trim(value: str) -> tuple[str, Paper3DAction]:
    name, raw = value.split("=", 1)
    alpha, beta, epsilon = [float(part.strip()) for part in raw.split(",")]
    return name.strip(), Paper3DAction(alpha=alpha, beta=beta, epsilon=epsilon)


def compare_formal(summary: dict) -> dict:
    learned = summary["frozen_attacker_learned_defender"]
    fixed = summary["frozen_attacker_fixed_trim"]
    no_defense = summary["frozen_attacker_no_defense"]
    return {
        "learned_acc_delta_vs_fixed_trim": float(learned["mean_post_clean_acc"])
        - float(fixed["mean_post_clean_acc"]),
        "learned_loss_delta_vs_fixed_trim": float(learned["mean_post_clean_loss"])
        - float(fixed["mean_post_clean_loss"]),
        "fixed_trim_acc_delta_vs_no_defense": float(fixed["mean_post_clean_acc"])
        - float(no_defense["mean_post_clean_acc"]),
        "fixed_trim_loss_delta_vs_no_defense": float(fixed["mean_post_clean_loss"])
        - float(no_defense["mean_post_clean_loss"]),
    }


if __name__ == "__main__":
    main()
