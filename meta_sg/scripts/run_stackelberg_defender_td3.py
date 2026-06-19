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
    parser.add_argument("--train-mode", choices=["episodic", "continuous_online"], default="episodic")
    parser.add_argument("--total-train-steps", type=int, default=0)
    parser.add_argument("--online-update-interval", type=int, default=100)
    parser.add_argument("--updates-per-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-horizon", type=int, default=5)
    parser.add_argument("--eval-online-adapt", action="store_true")
    parser.add_argument("--eval-online-update-interval", type=int, default=100)
    parser.add_argument("--eval-online-updates-per-interval", type=int, default=100)
    parser.add_argument("--eval-online-exploration-noise", type=float, default=0.0)
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
    parser.add_argument("--third-action", choices=["neuroclip", "server_lr"], default="neuroclip")
    parser.add_argument("--server-lr-min", type=float, default=0.0)
    parser.add_argument("--server-lr-max", type=float, default=1.0)
    parser.add_argument("--fixed-paper3d-baselines", nargs="*", default=None)
    parser.add_argument("--skip-fixed-baselines", action="store_true")
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
    parser.add_argument("--nonfinite-loss-penalty", type=float, default=1000.0)
    parser.add_argument("--acc-floor", type=float, default=0.0)
    parser.add_argument("--acc-floor-penalty-weight", type=float, default=0.0)
    parser.add_argument("--malicious-norm-threshold", type=float, default=0.0)
    parser.add_argument("--malicious-norm-penalty-weight", type=float, default=0.0)
    parser.add_argument("--reward-clip-min", type=float, default=0.0)
    parser.add_argument("--continuous-safe-action", nargs=3, type=float, default=None)
    parser.add_argument("--safe-fallback-acc-threshold", type=float, default=0.0)
    parser.add_argument("--safe-fallback-loss-threshold", type=float, default=0.0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--stop-at-step", type=int, default=0)
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
    lock_path = acquire_run_lock(output_dir)
    writer = None
    try:
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
        resume_rows: list[dict] | None = None
        resume_global_step = 0
        resume_episode = 0
        resume_snapshot = None
        if str(args.resume_checkpoint or ""):
            resume_rows, resume_global_step, resume_episode, resume_snapshot = load_training_checkpoint(
                Path(args.resume_checkpoint),
                defender=defender,
                buffer=buffer,
            )

        train_rows = select_train_defender(
            args,
            env,
            defender,
            buffer,
            writer=writer,
            warmup_weights=warmup_weights,
            output_dir=output_dir,
            initial_rows=resume_rows,
            initial_global_step=resume_global_step,
            start_episode=resume_episode,
            resume_snapshot=resume_snapshot,
        )
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
        online_adapt_summary = {}
        if bool(args.eval_online_adapt):
            online_buffer = ReplayBuffer(capacity=int(args.buffer_capacity), obs_dim=int(obs.shape[0]), act_dim=3)
            online_rows = evaluate_defender_online_adapt(
                args,
                build_env(
                    args,
                    attacker_checkpoint=attacker_checkpoint,
                    distribution_dir=distribution_dir,
                    seed=int(args.seed) + 20_000,
                    horizon=int(args.eval_horizon or args.horizon),
                ),
                defender,
                online_buffer,
                writer=writer,
                warmup_weights=warmup_weights,
            )
            online_dir = output_dir / "eval_online_adapt"
            write_rows(online_dir / "rounds.csv", online_rows)
            online_adapt_summary = summarize_rows(online_rows, online_dir)
        baselines: dict[str, dict] = {}
        baseline_actions = {} if bool(args.skip_fixed_baselines) else parse_fixed_paper3d_baselines(args.fixed_paper3d_baselines)
        for name, action in baseline_actions.items():
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
            "train_mode": str(args.train_mode),
            "total_train_steps": int(args.total_train_steps or (args.episodes * args.horizon)),
            "online_update_interval": int(args.online_update_interval),
            "updates_per_interval": int(args.updates_per_interval),
            "train": summarize_rows(train_rows, output_dir),
            "eval_learned": learned_summary,
            "eval_online_adapt": online_adapt_summary,
            "fixed_paper3d_baselines": baselines,
            "comparison": compare_learned_to_fixed_paper3d(learned_summary, baselines),
            "reward_scale": float(args.reward_scale),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log_eval_summaries(writer, learned_summary, baselines)
        print(
            "STACKELBERG_PAPER3D_DEFENDER_TD3_RESULT",
            f"steps={len(train_rows)}",
            f"eval_mean_reward={learned_summary['mean_defender_reward']:.6f}",
            f"eval_mean_post_clean_loss={learned_summary['mean_post_clean_loss']:.6f}",
            f"output={output_dir}",
        )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        release_run_lock(lock_path)


def acquire_run_lock(output_dir: Path) -> Path:
    lock_path = output_dir / "training.lock"
    if lock_path.exists():
        text = lock_path.read_text(encoding="utf-8").strip()
        try:
            pid = int(text)
        except ValueError:
            pid = -1
        if pid > 0 and _pid_is_alive(pid):
            raise RuntimeError(f"run already running with pid={pid}: {lock_path}")
        lock_path.unlink()
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is not None and lock_path.exists():
        lock_path.unlink()


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
    warmup_checkpoint = output_dir / "warmup_checkpoint.pt"
    if str(getattr(args, "resume_checkpoint", "") or "") and warmup_checkpoint.exists():
        checkpoint = torch.load(warmup_checkpoint, map_location="cpu", weights_only=False)
        return checkpoint["weights"], dict(checkpoint.get("metrics", {}))

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
            third_action=str(args.third_action),
            server_lr_min=float(args.server_lr_min),
            server_lr_max=float(args.server_lr_max),
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
    output_dir: Path | None = None,
    initial_rows: list[dict] | None = None,
    initial_global_step: int = 0,
    start_episode: int = 0,
) -> list[dict]:
    rows: list[dict] = list(initial_rows or [])
    global_step = int(initial_global_step)
    stop_at_step = int(getattr(args, "stop_at_step", 0) or 0)
    if stop_at_step > 0 and global_step >= stop_at_step:
        return rows
    for episode in range(int(start_episode), int(args.episodes)):
        if warmup_weights is None:
            obs = env.reset(seed=int(args.seed) + episode)
        else:
            obs = reset_env_from_weights(env, seed=int(args.seed) + episode, weights=warmup_weights)
        episode_return = 0.0
        for _ in range(int(args.horizon)):
            raw_action = choose_training_action(args, defender, obs, global_step=global_step)
            next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
            defender_reward = compute_defender_reward(args, info)
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
            if output_dir is not None and _should_checkpoint(args, global_step):
                save_latest_training_checkpoint(
                    output_dir,
                    args=args,
                    defender=defender,
                    buffer=buffer,
                    rows=rows,
                    global_step=global_step,
                    episode=episode,
                )
            obs = next_obs
            if stop_at_step > 0 and global_step >= stop_at_step:
                return rows
            if done:
                break
    return rows


def select_train_defender(
    args: argparse.Namespace,
    env: BSMGEnv,
    defender,
    buffer,
    *,
    writer=None,
    warmup_weights: list[np.ndarray] | None = None,
    output_dir: Path | None = None,
    initial_rows: list[dict] | None = None,
    initial_global_step: int = 0,
    start_episode: int = 0,
    resume_snapshot=None,
) -> list[dict]:
    if str(getattr(args, "train_mode", "episodic")) == "continuous_online":
        return train_defender_continuous_online(
            args,
            env,
            defender,
            buffer,
            writer=writer,
            warmup_weights=warmup_weights,
            output_dir=output_dir,
            initial_rows=initial_rows,
            initial_global_step=initial_global_step,
            resume_snapshot=resume_snapshot,
        )
    return train_defender(
        args,
        env,
        defender,
        buffer,
        writer=writer,
        warmup_weights=warmup_weights,
        output_dir=output_dir,
        initial_rows=initial_rows,
        initial_global_step=initial_global_step,
        start_episode=start_episode,
    )


def train_defender_continuous_online(
    args: argparse.Namespace,
    env: BSMGEnv,
    defender,
    buffer,
    *,
    writer=None,
    warmup_weights: list[np.ndarray] | None = None,
    output_dir: Path | None = None,
    initial_rows: list[dict] | None = None,
    initial_global_step: int = 0,
    resume_snapshot=None,
) -> list[dict]:
    rows: list[dict] = list(initial_rows or [])
    global_step = int(initial_global_step)
    total_steps = int(getattr(args, "total_train_steps", 0) or 0)
    if total_steps <= 0:
        total_steps = int(args.episodes) * int(args.horizon)
    stop_at_step = int(getattr(args, "stop_at_step", 0) or 0)
    if stop_at_step > 0:
        total_steps = min(total_steps, stop_at_step)
    if global_step >= total_steps:
        return rows

    if resume_snapshot is not None:
        obs = restore_env_training_snapshot(env, resume_snapshot)
    elif warmup_weights is None:
        obs = env.reset(seed=int(args.seed))
    else:
        obs = reset_env_from_weights(env, seed=int(args.seed), weights=warmup_weights)

    episode_return = 0.0
    episode = 0
    interval = max(1, int(getattr(args, "online_update_interval", 1) or 1))
    updates_per_interval = max(0, int(getattr(args, "updates_per_interval", 0) or 0))
    previous_info: dict | None = None
    while global_step < total_steps:
        raw_action, used_safe_fallback = choose_continuous_training_action(
            args,
            defender,
            obs,
            global_step=global_step,
            previous_info=previous_info,
        )
        next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
        defender_reward = compute_defender_reward(args, info)
        buffer.add(obs, raw_action, defender_reward, next_obs, False)
        global_step += 1
        episode_return += float(defender_reward)

        stats = {}
        if global_step % interval == 0:
            for _ in range(updates_per_interval):
                stats = defender.update(buffer)

        info_for_row = dict(info)
        local_round = int(info_for_row.get("round", global_step))
        info_for_row["round"] = global_step
        row = make_round_row(
            global_step=global_step,
            episode=episode,
            info=info_for_row,
            raw_action=raw_action,
            action=raw_action_to_paper3d(raw_action, args),
            defender_reward=defender_reward,
            attacker_reward=attacker_reward,
            episode_return=episode_return,
            stats=stats,
        )
        row["local_round"] = local_round
        row["safe_fallback"] = int(bool(used_safe_fallback))
        rows.append(row)
        log_train_row(writer, row, global_step)
        _print_progress(args, "train_continuous", row)
        obs = next_obs
        previous_info = dict(info)

        if done:
            continue_env_after_horizon(env)
        if output_dir is not None and _should_checkpoint(args, global_step):
            save_latest_training_checkpoint(
                output_dir,
                args=args,
                defender=defender,
                buffer=buffer,
                rows=rows,
                global_step=global_step,
                episode=episode,
                env=env,
                obs=obs,
            )

    return rows


def continue_env_after_horizon(env: BSMGEnv) -> None:
    if hasattr(env, "_round"):
        env._round = 0
    if hasattr(env, "_history"):
        env._history = []


def capture_env_training_snapshot(env: BSMGEnv, obs: np.ndarray | None = None) -> dict:
    snapshot = env.coordinator.snapshot()
    return {
        "coordinator_round_idx": int(snapshot.round_idx),
        "weights": clone_weights(snapshot.weights),
        "env_round": int(getattr(env, "_round", 0)),
        "obs": None if obs is None else np.asarray(obs, dtype=np.float32).copy(),
    }


def restore_env_training_snapshot(env: BSMGEnv, payload: dict) -> np.ndarray:
    from meta_sg.simulation.types import SimulationSnapshot

    env.coordinator.restore(
        SimulationSnapshot(
            round_idx=int(payload.get("coordinator_round_idx", 0)),
            weights=clone_weights(payload["weights"]),
            rng_state=None,
        )
    )
    env_round = int(payload.get("env_round", 0))
    horizon = int(getattr(getattr(env, "config", None), "horizon", 0) or 0)
    if horizon > 0 and env_round >= horizon:
        env_round = 0
    env._round = env_round
    if hasattr(env, "_history"):
        env._history = []
    obs = payload.get("obs")
    if obs is None:
        obs = env._make_obs(env.coordinator.current_weights)
    obs = np.asarray(obs, dtype=np.float32)
    env._obs = obs
    env._obs_dim = obs.shape[0]
    return obs


def _should_checkpoint(args: argparse.Namespace, global_step: int) -> bool:
    interval = int(getattr(args, "checkpoint_every", 0) or 0)
    return interval > 0 and int(global_step) > 0 and int(global_step) % interval == 0


def save_latest_training_checkpoint(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    defender,
    buffer,
    rows: list[dict],
    global_step: int,
    episode: int,
    env: BSMGEnv | None = None,
    obs: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "global_step": int(global_step),
        "episode": int(episode),
        "next_episode": int(episode) + 1,
        "defender": defender.algorithm.state_dict(),
        "defender_step": int(getattr(defender, "_step", 0)),
        "algorithm_cnt": int(getattr(defender.algorithm, "_cnt", 0)),
        "algorithm_last": getattr(defender.algorithm, "_last", 0),
        "replay_buffer": compact_replay_buffer(buffer),
        "rows": list(rows),
        "args": vars(args),
        "env_snapshot": None if env is None else capture_env_training_snapshot(env, obs),
        "np_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    tmp_checkpoint_path = output_dir / "latest_checkpoint.pt.tmp"
    torch.save(checkpoint, tmp_checkpoint_path)
    tmp_checkpoint_path.replace(checkpoint_path)
    write_rows(output_dir / "rounds_partial.csv", rows)


def load_training_checkpoint(path: Path, *, defender, buffer) -> tuple[list[dict], int, int, dict | None]:
    checkpoint = torch.load(path.expanduser(), map_location=getattr(defender, "device", "cpu"), weights_only=False)
    defender.algorithm.load_state_dict(checkpoint["defender"])
    defender._step = int(checkpoint.get("defender_step", 0))
    defender.algorithm._cnt = int(checkpoint.get("algorithm_cnt", defender.algorithm._cnt))
    defender.algorithm._last = checkpoint.get("algorithm_last", defender.algorithm._last)
    replay_buffer = checkpoint.get("replay_buffer")
    if replay_buffer is not None:
        restore_replay_buffer(buffer, replay_buffer)
    if "np_random_state" in checkpoint:
        np.random.set_state(checkpoint["np_random_state"])
    if "torch_random_state" in checkpoint:
        torch_random_state = checkpoint["torch_random_state"]
        if hasattr(torch_random_state, "cpu"):
            torch_random_state = torch_random_state.cpu()
        torch.set_rng_state(torch_random_state)
    rows = list(checkpoint.get("rows", []))
    return (
        rows,
        int(checkpoint.get("global_step", len(rows))),
        int(checkpoint.get("next_episode", 0)),
        checkpoint.get("env_snapshot"),
    )


def compact_replay_buffer(buffer) -> dict | None:
    if buffer is None:
        return None
    source = buffer.tianshou_buffer
    size = len(source)
    return {
        "capacity": int(buffer.capacity),
        "obs_dim": int(buffer.obs_dim),
        "act_dim": int(buffer.act_dim),
        "size": int(size),
        "obs": np.asarray(source.obs[:size], dtype=np.float32).copy(),
        "act": np.asarray(source.act[:size], dtype=np.float32).copy(),
        "rew": np.asarray(source.rew[:size], dtype=np.float32).copy(),
        "obs_next": np.asarray(source.obs_next[:size], dtype=np.float32).copy(),
        "done": np.asarray(source.done[:size], dtype=np.bool_).copy(),
    }


def restore_replay_buffer(buffer, payload: dict) -> None:
    buffer._buffer.reset()
    size = int(payload.get("size", 0))
    for idx in range(size):
        buffer.add(
            payload["obs"][idx],
            payload["act"][idx],
            float(payload["rew"][idx]),
            payload["obs_next"][idx],
            bool(payload["done"][idx]),
        )


def choose_training_action(args: argparse.Namespace, defender, obs: np.ndarray, *, global_step: int) -> np.ndarray:
    if int(global_step) < int(args.warmup_steps):
        return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
    return defender.get_action(obs, noise=float(args.exploration_noise))


def choose_continuous_training_action(
    args: argparse.Namespace,
    defender,
    obs: np.ndarray,
    *,
    global_step: int,
    previous_info: dict | None = None,
) -> tuple[np.ndarray, bool]:
    if _should_use_safe_fallback(args, previous_info):
        return continuous_safe_raw_action(args), True
    return choose_training_action(args, defender, obs, global_step=global_step), False


def _should_use_safe_fallback(args: argparse.Namespace, previous_info: dict | None) -> bool:
    if previous_info is None:
        return False
    acc_threshold = float(getattr(args, "safe_fallback_acc_threshold", 0.0) or 0.0)
    if acc_threshold > 0.0 and _info_float(previous_info, "post_clean_acc", default=1.0) < acc_threshold:
        return True
    loss_threshold = float(getattr(args, "safe_fallback_loss_threshold", 0.0) or 0.0)
    loss = _info_float(previous_info, "post_clean_loss", default=0.0)
    return loss_threshold > 0.0 and (not np.isfinite(loss) or loss > loss_threshold)


def continuous_safe_raw_action(args: argparse.Namespace) -> np.ndarray:
    values = getattr(args, "continuous_safe_action", None)
    if values is None:
        action = Paper3DAction(
            alpha=float(args.alpha_min),
            beta=float(args.beta_max),
            epsilon=float(args.neuroclip_eps_min),
        )
    else:
        action = Paper3DAction(alpha=float(values[0]), beta=float(values[1]), epsilon=float(values[2]))
    return fixed_paper3d_to_raw_action(action, args)


def compute_defender_reward(args: argparse.Namespace, info: dict) -> float:
    reward_info = dict(info)
    loss = _info_float(reward_info, "post_clean_loss", default=float("nan"))
    if not np.isfinite(loss):
        reward_info["post_clean_loss"] = float(getattr(args, "nonfinite_loss_penalty", 1000.0))
    reward = compute_paper_defender_reward(reward_info, scale=float(args.reward_scale))

    acc_floor = float(getattr(args, "acc_floor", 0.0) or 0.0)
    acc_floor_weight = float(getattr(args, "acc_floor_penalty_weight", 0.0) or 0.0)
    if acc_floor_weight > 0.0 and acc_floor > 0.0:
        post_acc = _info_float(reward_info, "post_clean_acc", default=0.0)
        reward -= acc_floor_weight * max(0.0, acc_floor - post_acc)

    norm_weight = float(getattr(args, "malicious_norm_penalty_weight", 0.0) or 0.0)
    if norm_weight > 0.0:
        threshold = float(getattr(args, "malicious_norm_threshold", 0.0) or 0.0)
        reward -= norm_weight * max(0.0, _mean_malicious_norm(reward_info) - threshold)

    clip_min = float(getattr(args, "reward_clip_min", 0.0) or 0.0)
    if clip_min < 0.0:
        reward = max(float(reward), clip_min)
    return float(reward)


def _info_float(info: dict, key: str, *, default: float) -> float:
    try:
        value = float(info.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return value


def _mean_malicious_norm(info: dict) -> float:
    if "malicious_update_norms" in info:
        norms = np.asarray(info.get("malicious_update_norms", []), dtype=np.float32)
        if norms.size:
            value = float(np.mean(norms))
            return value if np.isfinite(value) else 0.0
    value = _info_float(info, "malicious_update_norm", default=0.0)
    return value if np.isfinite(value) else 0.0


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
            defender_reward = compute_defender_reward(args, info)
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


def evaluate_defender_online_adapt(
    args: argparse.Namespace,
    env,
    defender,
    buffer,
    *,
    writer=None,
    warmup_weights: list[np.ndarray] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    global_step = 0
    episode_return = 0.0
    horizon = int(args.eval_horizon or args.horizon)
    interval = max(1, int(getattr(args, "eval_online_update_interval", 1) or 1))
    updates_per_interval = max(0, int(getattr(args, "eval_online_updates_per_interval", 0) or 0))
    if warmup_weights is None:
        obs = env.reset(seed=int(args.seed) + 10_000)
    else:
        obs = reset_env_from_weights(env, seed=int(args.seed) + 10_000, weights=warmup_weights)
    previous_info: dict | None = None
    for _ in range(horizon):
        if _should_use_safe_fallback(args, previous_info):
            raw_action = continuous_safe_raw_action(args)
            used_safe_fallback = True
        else:
            raw_action = defender.get_action(obs, noise=float(args.eval_online_exploration_noise))
            used_safe_fallback = False
        next_obs, _, attacker_reward, done, info = env.step(raw_action, np.zeros(3, dtype=np.float32))
        defender_reward = compute_defender_reward(args, info)
        buffer.add(obs, raw_action, defender_reward, next_obs, False)
        global_step += 1
        episode_return += float(defender_reward)

        stats = {}
        if global_step % interval == 0:
            for _ in range(updates_per_interval):
                stats = defender.update(buffer)

        row = make_round_row(
            global_step=global_step,
            episode=0,
            info=info,
            raw_action=raw_action,
            action=raw_action_to_paper3d(raw_action, args),
            defender_reward=defender_reward,
            attacker_reward=attacker_reward,
            episode_return=episode_return,
            stats=stats,
        )
        row["online_adapt"] = 1
        row["safe_fallback"] = int(bool(used_safe_fallback))
        rows.append(row)
        log_train_row(writer, row, global_step)
        _print_progress(args, "eval_online_adapt", row)
        obs = next_obs
        previous_info = dict(info)
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
            defender_reward = compute_defender_reward(args, info)
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
