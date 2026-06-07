"""Train a paper-aligned 3D TD3 defender against a frozen fl_sandbox RL attacker."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.stackelberg.metrics import (
    compare_learned_to_fixed_paper3d,
    format_progress_line,
    make_round_row,
    summarize_rows,
    write_rows,
)
from meta_sg.stackelberg.paper3d_action import (
    Paper3DAction,
    fixed_paper3d_to_raw_action,
    parse_fixed_paper3d_baselines,
    raw_action_to_paper3d,
)
from meta_sg.stackelberg.paper3d_sandbox import FrozenSandboxAttack, Paper3DSandboxCoordinator
from meta_sg.stackelberg.paper_reward import compute_paper_defender_reward
from meta_sg.stackelberg.tensorboard import create_summary_writer, log_eval_summaries, log_train_row
from meta_sg.stackelberg.warmup import clone_weights, reset_env_from_weights
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT = (
    REPO_ROOT
    / "fl_sandbox/outputs/repro_7faf353_currentdist_20260602"
    / "mnist_rl_clipped_median_paper_q_q0.1_500r/checkpoints/rl_policy_latest.pt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="runs/stackelberg_paper3d_defender_td3")
    parser.add_argument("--attacker-checkpoint", default="")
    parser.add_argument("--rl-distribution-dir", default="")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-horizon", type=int, default=5)
    parser.add_argument("--clean-warmup-rounds", type=int, default=100)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=10_000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--initial-raw-action", nargs=3, type=float, default=[-0.95, -0.111111, 1.0])
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=30.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=0.45)
    parser.add_argument("--neuroclip-eps-min", type=float, default=2.0)
    parser.add_argument("--neuroclip-eps-max", type=float, default=10.0)
    parser.add_argument("--fixed-paper3d-baselines", nargs="*", default=None)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", type=float, default=0.1)
    parser.add_argument("--client-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    attacker_checkpoint = resolve_attacker_checkpoint(args.attacker_checkpoint)
    distribution_dir = resolve_distribution_dir(args.rl_distribution_dir, attacker_checkpoint)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / f"fl_sandbox_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(output_dir / "tensorboard", enabled=bool(args.tensorboard))

    warmup_weights, warmup_metrics = run_clean_warmup(args, output_dir=output_dir)
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed),
    )
    obs = reset_env_from_weights(env, seed=int(args.seed), weights=warmup_weights)

    from meta_sg.learning.config import TD3Config
    from meta_sg.learning.replay_buffer import ReplayBuffer
    from meta_sg.learning.td3 import TD3Agent

    cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=int(args.warmup_steps),
        exploration_noise=float(args.exploration_noise),
    )
    defender = TD3Agent(obs_dim=int(obs.shape[0]), act_dim=3, config=cfg, device=torch.device(args.device))
    if args.initial_raw_action is not None:
        initialize_defender_actor_constant3(
            defender,
            raw_action=np.asarray(args.initial_raw_action, dtype=np.float32),
        )
    buffer = ReplayBuffer(capacity=int(args.buffer_capacity), obs_dim=int(obs.shape[0]), act_dim=3)

    train_rows = train_defender(args, env, defender, buffer, writer=writer, warmup_weights=warmup_weights)
    write_rows(output_dir / "rounds.csv", train_rows)
    defender.save(str(output_dir / "defender_td3.pt"))

    eval_rows = evaluate_defender(
        args,
        defender,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        warmup_weights=warmup_weights,
    )
    eval_dir = output_dir / "eval_learned"
    write_rows(eval_dir / "rounds.csv", eval_rows)

    learned_summary = summarize_rows(eval_rows, eval_dir)
    baselines: dict[str, dict] = {}
    for name, action in parse_fixed_paper3d_baselines(args.fixed_paper3d_baselines).items():
        baseline_rows = evaluate_fixed_paper3d(
            args,
            action,
            attacker_checkpoint=attacker_checkpoint,
            distribution_dir=distribution_dir,
            warmup_weights=warmup_weights,
        )
        baseline_dir = output_dir / f"baseline_{name}"
        write_rows(baseline_dir / "rounds.csv", baseline_rows)
        baselines[name] = summarize_rows(baseline_rows, baseline_dir)

    summary = {
        "attacker_checkpoint": str(attacker_checkpoint),
        "rl_distribution_dir": str(distribution_dir),
        "clean_warmup_rounds": int(args.clean_warmup_rounds),
        "warmup_metrics": warmup_metrics,
        "train": summarize_rows(train_rows, output_dir),
        "eval_learned": learned_summary,
        "fixed_paper3d_baselines": baselines,
        "comparison": compare_learned_to_fixed_paper3d(learned_summary, baselines),
        "reward_scale": float(args.reward_scale),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_eval_summaries(writer, learned_summary, baselines)
    if writer is not None:
        writer.flush()
        writer.close()
    print(
        "STACKELBERG_PAPER3D_DEFENDER_TD3_RESULT",
        f"steps={len(train_rows)}",
        f"eval_mean_reward={learned_summary['mean_defender_reward']:.6f}",
        f"eval_mean_post_clean_loss={learned_summary['mean_post_clean_loss']:.6f}",
        f"output={output_dir}",
    )


def resolve_attacker_checkpoint(path: str | Path) -> Path:
    resolved = Path(path).expanduser() if str(path or "") else DEFAULT_PRETRAINED_ATTACKER_CHECKPOINT
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"attacker checkpoint not found: {resolved}")
    return resolved


def resolve_distribution_dir(path: str | Path, attacker_checkpoint: Path) -> Path:
    if str(path or ""):
        resolved = Path(path).expanduser()
    else:
        checkpoint = torch.load(attacker_checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        resolved = Path(str(checkpoint_config.get("rl_distribution_dir", ""))).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"rl distribution dir not found: {resolved}")
    return resolved


def run_clean_warmup(args: argparse.Namespace, *, output_dir: Path) -> tuple[list[np.ndarray], dict[str, float]]:
    config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=str(args.device),
        attack_type="clean",
        defense_type="fedavg",
        rounds=int(args.clean_warmup_rounds),
        start_round_idx=1,
        warmup_rounds=0,
        num_clients=int(args.num_clients),
        num_attackers=int(args.num_attackers),
        subsample_rate=float(args.subsample_rate),
        local_epochs=1,
        batch_size=int(args.batch_size_fl),
        eval_batch_size=int(args.eval_batch_size),
        fltrust_root_size=0,
        max_client_samples_per_client=int(args.client_samples),
        max_eval_samples=int(args.eval_samples),
        seed=int(args.seed) + 1_000,
    )
    coordinator = FLSandboxCoordinatorAdapter(config)
    coordinator.reset(seed=int(args.seed) + 1_000)
    metrics: dict[str, float] = {}
    for round_idx in range(1, int(args.clean_warmup_rounds) + 1):
        summary = coordinator.runner.run_round(round_idx, attack=None, evaluate=True)
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
    weights = clone_weights(coordinator.current_weights)
    torch.save({"weights": weights, "metrics": metrics}, output_dir / "warmup_checkpoint.pt")
    return weights, metrics


def initialize_defender_actor_constant3(defender, *, raw_action: np.ndarray) -> None:
    raw = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(-1), -0.999, 0.999)
    if raw.shape[0] != 3:
        raise ValueError("paper3d defender initial raw action must have shape (3,)")
    bias = np.arctanh(raw).astype(np.float32)
    with torch.no_grad():
        last = defender.actor.last.model[0]
        last.weight.zero_()
        last.bias.copy_(torch.as_tensor(bias, dtype=last.bias.dtype, device=last.bias.device))
    defender.algorithm._lagged_networks.full_parameter_update()


def build_env(
    args: argparse.Namespace,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    seed: int,
    horizon: int | None = None,
) -> BSMGEnv:
    config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=str(args.device),
        attack_type="rl",
        defense_type="paper_norm_trimmed_mean",
        rounds=int(horizon or args.horizon),
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
        seed=int(seed),
    )
    return BSMGEnv(
        coordinator=Paper3DSandboxCoordinator(config),
        attack_type=ATTACK_DOMAIN["rl"],
        attack_strategy=FrozenSandboxAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=int(horizon or args.horizon),
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
            beta_min=float(args.beta_min),
            beta_max=float(args.beta_max),
            eps_min=float(args.neuroclip_eps_min),
            eps_max=float(args.neuroclip_eps_max),
            use_neuroclip=True,
            reward_mode="loss",
            lambda_bd=0.0,
            eval_every=int(args.eval_every),
        ),
    )


def train_defender(
    args: argparse.Namespace,
    env: BSMGEnv,
    defender,
    buffer,
    *,
    writer=None,
    warmup_weights: list[np.ndarray] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    global_step = 0
    for episode in range(int(args.episodes)):
        if warmup_weights is None:
            obs = env.reset(seed=int(args.seed) + episode)
        else:
            obs = reset_env_from_weights(env, seed=int(args.seed) + episode, weights=warmup_weights)
        episode_return = 0.0
        for _ in range(int(args.horizon)):
            raw_action = choose_training_action(args, defender, obs, global_step=global_step)
            next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
            defender_reward = compute_paper_defender_reward(info, scale=float(args.reward_scale))
            buffer.add(obs, raw_action, defender_reward, next_obs, done)
            stats = {}
            for _ in range(int(args.updates_per_step)):
                stats = defender.update(buffer)
            global_step += 1
            episode_return += float(defender_reward)
            row = make_round_row(
                global_step=global_step,
                episode=episode,
                info=info,
                raw_action=raw_action,
                action=raw_action_to_paper3d(raw_action, args),
                defender_reward=defender_reward,
                attacker_reward=attacker_reward,
                episode_return=episode_return,
                stats=stats,
            )
            rows.append(row)
            log_train_row(writer, row, global_step)
            _print_progress(args, "train", row)
            obs = next_obs
            if done:
                break
    return rows


def choose_training_action(args: argparse.Namespace, defender, obs: np.ndarray, *, global_step: int) -> np.ndarray:
    if int(global_step) < int(args.warmup_steps):
        return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
    return defender.get_action(obs, noise=float(args.exploration_noise))


def evaluate_defender(
    args: argparse.Namespace,
    defender,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    warmup_weights: list[np.ndarray] | None = None,
) -> list[dict]:
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed) + 10_000,
        horizon=int(args.eval_horizon or args.horizon),
    )
    rows: list[dict] = []
    global_step = 0
    for episode in range(int(args.eval_episodes)):
        if warmup_weights is None:
            obs = env.reset(seed=int(args.seed) + 10_000 + episode)
        else:
            obs = reset_env_from_weights(env, seed=int(args.seed) + 10_000 + episode, weights=warmup_weights)
        episode_return = 0.0
        for _ in range(int(args.eval_horizon or args.horizon)):
            raw_action = defender.get_action(obs, noise=0.0)
            next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
            defender_reward = compute_paper_defender_reward(info, scale=float(args.reward_scale))
            global_step += 1
            episode_return += float(defender_reward)
            row = make_round_row(
                global_step=global_step,
                episode=episode,
                info=info,
                raw_action=raw_action,
                action=raw_action_to_paper3d(raw_action, args),
                defender_reward=defender_reward,
                attacker_reward=attacker_reward,
                episode_return=episode_return,
                stats={},
            )
            rows.append(row)
            _print_progress(args, "eval_learned", row)
            obs = next_obs
            if done:
                break
    return rows


def evaluate_fixed_paper3d(
    args: argparse.Namespace,
    action: Paper3DAction,
    *,
    attacker_checkpoint: Path,
    distribution_dir: Path,
    warmup_weights: list[np.ndarray] | None = None,
) -> list[dict]:
    env = build_env(
        args,
        attacker_checkpoint=attacker_checkpoint,
        distribution_dir=distribution_dir,
        seed=int(args.seed) + 10_000,
        horizon=int(args.eval_horizon or args.horizon),
    )
    raw_action = fixed_paper3d_to_raw_action(action, args)
    rows: list[dict] = []
    global_step = 0
    for episode in range(int(args.eval_episodes)):
        if warmup_weights is None:
            obs = env.reset(seed=int(args.seed) + 10_000 + episode)
        else:
            obs = reset_env_from_weights(env, seed=int(args.seed) + 10_000 + episode, weights=warmup_weights)
        del obs
        episode_return = 0.0
        for _ in range(int(args.eval_horizon or args.horizon)):
            _, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
            defender_reward = compute_paper_defender_reward(info, scale=float(args.reward_scale))
            global_step += 1
            episode_return += float(defender_reward)
            row = make_round_row(
                global_step=global_step,
                episode=episode,
                info=info,
                raw_action=raw_action,
                action=action,
                defender_reward=defender_reward,
                attacker_reward=attacker_reward,
                episode_return=episode_return,
                stats={},
            )
            rows.append(row)
            _print_progress(args, "eval_fixed", row)
            if done:
                break
    return rows


def _print_progress(args: argparse.Namespace, phase: str, row: dict) -> None:
    interval = max(0, int(getattr(args, "print_every", 0) or 0))
    if interval <= 0:
        return
    step = int(row["step"])
    if step == 1 or step % interval == 0:
        print(format_progress_line(phase, row), flush=True)


if __name__ == "__main__":
    main()
