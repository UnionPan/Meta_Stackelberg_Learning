"""Compare 3D defender policies against one shared attacker on fl_sandbox."""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.config import TD3Config
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.interface import FLCoordinator
from meta_sg.strategies.attacks.adaptive import AdaptiveAttackStrategy
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN, DefenseDecision


@dataclass(frozen=True)
class DefenderSpec:
    name: str
    raw_action: np.ndarray


DEFENDER_SPECS = (
    DefenderSpec("fixed_mid_defender", np.asarray([0.0, 0.0, 0.0], dtype=np.float32)),
    DefenderSpec("fixed_strong_clip_defender", np.asarray([-0.8, 0.2, 0.0], dtype=np.float32)),
    DefenderSpec("learned_meta_defender", np.zeros(3, dtype=np.float32)),
    DefenderSpec("colearning_defender", np.zeros(3, dtype=np.float32)),
)


@dataclass(frozen=True)
class DefenderActionConfig:
    alpha_min: float = 0.0
    alpha_max: float = 5.0
    beta_min: float = 0.0
    beta_max: float = 0.45
    eps_min: float = 1.0
    eps_max: float = 10.0
    action_prior_weight: float = 0.0
    relative_alpha: bool = False
    reward_mode: str = "accuracy"


def safe_defender_config() -> DefenderActionConfig:
    return DefenderActionConfig(
        alpha_min=0.5,
        alpha_max=5.0,
        beta_min=0.0,
        beta_max=0.4,
        eps_min=1.0,
        eps_max=10.0,
        action_prior_weight=0.02,
    )


def paper_aligned_defender_config() -> DefenderActionConfig:
    return DefenderActionConfig(
        alpha_min=1e-6,
        alpha_max=5.0,
        beta_min=0.0,
        beta_max=0.45,
        eps_min=1.0,
        eps_max=10.0,
        action_prior_weight=0.0,
        relative_alpha=True,
        reward_mode="loss",
    )


def decode_defender_raw_action(
    raw: np.ndarray,
    *,
    config: DefenderActionConfig | None = None,
) -> DefenseDecision:
    cfg = config or DefenderActionConfig()
    return DefenseDecision.from_raw(
        np.asarray(raw, dtype=np.float32),
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        eps_min=cfg.eps_min,
        eps_max=cfg.eps_max,
    )


def total_defender_comparison_training_rounds(
    *,
    shared_attacker_episodes: int,
    learned_meta_iters: int,
    colearning_iters: int,
    defender_count: int,
    seeds: int,
    train_horizon: int,
    eval_horizon: int,
    colearning_post_br_defender_updates: int,
) -> int:
    colearning_rollouts = 1 + (1 if colearning_post_br_defender_updates > 0 else 0)
    return (
        int(shared_attacker_episodes) * int(train_horizon)
        + int(learned_meta_iters) * int(train_horizon)
        + int(colearning_iters) * int(train_horizon) * colearning_rollouts
        + int(defender_count) * int(seeds) * int(eval_horizon)
    )


def training_stage_round_offsets(
    *,
    shared_attacker_episodes: int,
    learned_meta_iters: int,
    train_horizon: int,
    colearning_post_br_defender_updates: int,
) -> dict[str, int]:
    del colearning_post_br_defender_updates
    shared_rounds = int(shared_attacker_episodes) * int(train_horizon)
    learned_rounds = int(learned_meta_iters) * int(train_horizon)
    return {
        "shared_attacker": 0,
        "learned_meta_defender": shared_rounds,
        "colearning_defender": shared_rounds + learned_rounds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-root", default="runs/meta_sg_defender_compare")
    parser.add_argument("--train-horizon", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=10)
    parser.add_argument("--shared-attacker-episodes", type=int, default=80)
    parser.add_argument("--shared-attacker-updates", type=int, default=8)
    parser.add_argument("--learned-meta-iters", type=int, default=80)
    parser.add_argument("--learned-meta-updates", type=int, default=8)
    parser.add_argument("--colearning-iters", type=int, default=80)
    parser.add_argument("--colearning-defender-updates", type=int, default=8)
    parser.add_argument("--colearning-attacker-updates", type=int, default=8)
    parser.add_argument("--colearning-post-br-defender-updates", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--client-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--safe-defender-actions", action="store_true")
    parser.add_argument("--paper-aligned-defender", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_root) / f"fl_sandbox_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "defender_comparison_metrics.csv"

    coordinator_factory = _coordinator_factory(args)
    if args.paper_aligned_defender:
        defender_action_cfg = paper_aligned_defender_config()
    elif args.safe_defender_actions:
        defender_action_cfg = safe_defender_config()
    else:
        defender_action_cfg = DefenderActionConfig()
    obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights())
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.exploration_noise),
    )
    total_rounds = total_defender_comparison_training_rounds(
        shared_attacker_episodes=args.shared_attacker_episodes,
        learned_meta_iters=args.learned_meta_iters,
        colearning_iters=args.colearning_iters,
        defender_count=len(DEFENDER_SPECS),
        seeds=len(args.seeds),
        train_horizon=args.train_horizon,
        eval_horizon=args.eval_horizon,
        colearning_post_br_defender_updates=args.colearning_post_br_defender_updates,
    )
    print("LOG_DIR", log_dir)
    print("CSV", csv_path)
    print("TOTAL_FL_ROUNDS_INCLUDING_EVAL", total_rounds)
    print("SHARED_ATTACKER", "adaptive_rl", "act_dim=3", "budget_updates=", args.shared_attacker_updates)
    print("DEFENDER_ACTION", "3D", "alpha,beta,post")
    train_writer = SummaryWriter(str(log_dir / "train_process"))
    train_writer.add_scalar("config/total_fl_rounds_including_eval", float(total_rounds), 0)
    offsets = training_stage_round_offsets(
        shared_attacker_episodes=args.shared_attacker_episodes,
        learned_meta_iters=args.learned_meta_iters,
        train_horizon=args.train_horizon,
        colearning_post_br_defender_updates=args.colearning_post_br_defender_updates,
    )

    shared_attacker = TD3Agent(obs_dim, 3, td3_cfg)
    _train_shared_attacker(
        coordinator_factory=coordinator_factory,
        obs_dim=obs_dim,
        td3_cfg=td3_cfg,
        attacker=shared_attacker,
        episodes=args.shared_attacker_episodes,
        horizon=args.train_horizon,
        updates_per_episode=args.shared_attacker_updates,
        exploration_noise=args.exploration_noise,
        defender_action_cfg=defender_action_cfg,
        writer=train_writer,
        round_offset=offsets["shared_attacker"],
    )

    learned_meta_defender = TD3Agent(obs_dim, 3, td3_cfg)
    _train_defender_against_fixed_attacker(
        coordinator_factory=coordinator_factory,
        obs_dim=obs_dim,
        td3_cfg=td3_cfg,
        defender=learned_meta_defender,
        attacker=shared_attacker,
        episodes=args.learned_meta_iters,
        horizon=args.train_horizon,
        updates_per_episode=args.learned_meta_updates,
        exploration_noise=args.exploration_noise,
        defender_action_cfg=defender_action_cfg,
        writer=train_writer,
        round_offset=offsets["learned_meta_defender"],
    )

    colearning_defender = TD3Agent(obs_dim, 3, td3_cfg)
    _train_colearning_defender(
        coordinator_factory=coordinator_factory,
        obs_dim=obs_dim,
        td3_cfg=td3_cfg,
        defender=colearning_defender,
        episodes=args.colearning_iters,
        horizon=args.train_horizon,
        defender_updates=args.colearning_defender_updates,
        attacker_updates=args.colearning_attacker_updates,
        post_br_defender_updates=args.colearning_post_br_defender_updates,
        exploration_noise=args.exploration_noise,
        defender_action_cfg=defender_action_cfg,
        writer=train_writer,
        round_offset=offsets["colearning_defender"],
    )
    train_writer.flush()
    train_writer.close()

    policies = {
        "fixed_mid_defender": ConstantActionPolicy(DEFENDER_SPECS[0].raw_action),
        "fixed_strong_clip_defender": ConstantActionPolicy(DEFENDER_SPECS[1].raw_action),
        "learned_meta_defender": learned_meta_defender,
        "colearning_defender": colearning_defender,
    }
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "defender",
                "seed",
                "step",
                "clean_acc",
                "attack_success",
                "defender_reward",
                "attacker_reward",
                "alpha",
                "beta",
                "post",
            ],
        )
        writer.writeheader()
        for name, defender_policy in policies.items():
            tb = SummaryWriter(str(log_dir / name))
            _evaluate_defender(
                coordinator_factory=coordinator_factory,
                obs_dim=obs_dim,
                td3_cfg=td3_cfg,
                defender_name=name,
                defender_policy=defender_policy,
                attacker=shared_attacker,
                defender_action_cfg=defender_action_cfg,
                seeds=args.seeds,
                horizon=args.eval_horizon,
                csv_writer=writer,
                tb_writer=tb,
            )
            tb.flush()
            tb.close()
    print("DONE")


def _coordinator_factory(args: argparse.Namespace) -> Callable[[], FLCoordinator]:
    sandbox_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=str(args.device),
        num_clients=int(args.num_clients),
        num_attackers=int(args.num_attackers),
        subsample_rate=1.0,
        local_epochs=1,
        batch_size=int(args.batch_size_fl),
        eval_batch_size=int(args.eval_batch_size),
        fltrust_root_size=0,
        max_client_samples_per_client=int(args.client_samples),
        max_eval_samples=int(args.eval_samples),
    )
    return lambda: FLSandboxCoordinatorAdapter(sandbox_config)


def _build_env(
    coordinator_factory: Callable[[], FLCoordinator],
    attacker: TD3Agent,
    horizon: int,
    defender_action_cfg: DefenderActionConfig,
) -> BSMGEnv:
    attack_type = ATTACK_DOMAIN["rl"]
    return BSMGEnv(
        coordinator=coordinator_factory(),
        attack_type=attack_type,
        attack_strategy=AdaptiveAttackStrategy(attack_type, attacker),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=horizon,
            eval_every=1,
            alpha_min=defender_action_cfg.alpha_min,
            alpha_max=defender_action_cfg.alpha_max,
            beta_min=defender_action_cfg.beta_min,
            beta_max=defender_action_cfg.beta_max,
            eps_min=defender_action_cfg.eps_min,
            eps_max=defender_action_cfg.eps_max,
            action_prior_weight=defender_action_cfg.action_prior_weight,
            relative_alpha=defender_action_cfg.relative_alpha,
            reward_mode=defender_action_cfg.reward_mode,
        ),
    )


def _train_shared_attacker(
    *,
    coordinator_factory: Callable[[], FLCoordinator],
    obs_dim: int,
    td3_cfg: TD3Config,
    attacker: TD3Agent,
    episodes: int,
    horizon: int,
    updates_per_episode: int,
    exploration_noise: float,
    defender_action_cfg: DefenderActionConfig,
    writer: SummaryWriter,
    round_offset: int,
) -> None:
    buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    defender_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    defender = ConstantActionPolicy(DEFENDER_SPECS[0].raw_action)
    for episode in range(int(episodes)):
        env = _build_env(coordinator_factory, attacker, horizon, defender_action_cfg)
        collector = TrajectoryCollector(
            env=env,
            defender=defender,
            attacker=attacker,
            defender_buffer=defender_buffer,
            attacker_buffer=buffer,
            exploration_noise=exploration_noise,
            store_attacker=True,
        )
        traj = collector.collect(horizon, seed=10_000 + episode)
        losses = []
        for _ in range(int(updates_per_episode)):
            loss = attacker.update(buffer)
            if loss:
                losses.append(loss)
        global_step = int(round_offset) + (episode + 1) * int(horizon)
        _log_train_episode(
            writer,
            prefix="train/shared_attacker",
            traj=traj,
            losses=losses,
            buffer_size=len(buffer),
            global_step=global_step,
        )
        if (episode + 1) % max(1, int(episodes) // 5) == 0:
            print(f"SHARED_ATTACKER episode={episode + 1}/{episodes} buffer={len(buffer)}")


def _train_defender_against_fixed_attacker(
    *,
    coordinator_factory: Callable[[], FLCoordinator],
    obs_dim: int,
    td3_cfg: TD3Config,
    defender: TD3Agent,
    attacker: TD3Agent,
    episodes: int,
    horizon: int,
    updates_per_episode: int,
    exploration_noise: float,
    defender_action_cfg: DefenderActionConfig,
    writer: SummaryWriter,
    round_offset: int,
) -> None:
    defender_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    attacker_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    for episode in range(int(episodes)):
        env = _build_env(coordinator_factory, attacker, horizon, defender_action_cfg)
        collector = TrajectoryCollector(
            env=env,
            defender=defender,
            attacker=attacker,
            defender_buffer=defender_buffer,
            attacker_buffer=attacker_buffer,
            exploration_noise=exploration_noise,
            store_attacker=False,
        )
        traj = collector.collect(horizon, seed=20_000 + episode)
        losses = []
        for _ in range(int(updates_per_episode)):
            loss = defender.update(defender_buffer)
            if loss:
                losses.append(loss)
        global_step = int(round_offset) + (episode + 1) * int(horizon)
        _log_train_episode(
            writer,
            prefix="train/learned_meta_defender",
            traj=traj,
            losses=losses,
            buffer_size=len(defender_buffer),
            global_step=global_step,
        )
        if (episode + 1) % max(1, int(episodes) // 5) == 0:
            print(f"LEARNED_META_DEFENDER episode={episode + 1}/{episodes} buffer={len(defender_buffer)}")


def _train_colearning_defender(
    *,
    coordinator_factory: Callable[[], FLCoordinator],
    obs_dim: int,
    td3_cfg: TD3Config,
    defender: TD3Agent,
    episodes: int,
    horizon: int,
    defender_updates: int,
    attacker_updates: int,
    post_br_defender_updates: int,
    exploration_noise: float,
    defender_action_cfg: DefenderActionConfig,
    writer: SummaryWriter,
    round_offset: int,
) -> TD3Agent:
    attacker = TD3Agent(obs_dim, 3, td3_cfg)
    defender_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    attacker_buffer = ReplayBuffer(td3_cfg.buffer_capacity, obs_dim, 3)
    for episode in range(int(episodes)):
        env = _build_env(coordinator_factory, attacker, horizon, defender_action_cfg)
        collector = TrajectoryCollector(
            env=env,
            defender=defender,
            attacker=attacker,
            defender_buffer=defender_buffer,
            attacker_buffer=attacker_buffer,
            exploration_noise=exploration_noise,
            store_attacker=True,
        )
        traj = collector.collect(horizon, seed=30_000 + episode)
        def_losses = []
        atk_losses = []
        for _ in range(int(defender_updates)):
            loss = defender.update(defender_buffer)
            if loss:
                def_losses.append(loss)
        for _ in range(int(attacker_updates)):
            loss = attacker.update(attacker_buffer)
            if loss:
                atk_losses.append(loss)
        rollouts = 1
        if int(post_br_defender_updates) > 0:
            env2 = _build_env(coordinator_factory, attacker, horizon, defender_action_cfg)
            collector2 = TrajectoryCollector(
                env=env2,
                defender=defender,
                attacker=attacker,
                defender_buffer=defender_buffer,
                attacker_buffer=attacker_buffer,
                exploration_noise=exploration_noise,
                store_attacker=True,
            )
            traj2 = collector2.collect(horizon, seed=40_000 + episode)
            rollouts += 1
            for _ in range(int(post_br_defender_updates)):
                loss = defender.update(defender_buffer)
                if loss:
                    def_losses.append(loss)
            traj = _merge_trajectories(traj, traj2)
        global_step = int(round_offset) + (episode + 1) * int(horizon) * rollouts
        _log_train_episode(
            writer,
            prefix="train/colearning_defender",
            traj=traj,
            losses=def_losses,
            buffer_size=len(defender_buffer),
            global_step=global_step,
        )
        _log_loss_means(writer, "train/colearning_attacker", atk_losses, global_step)
        writer.add_scalar("train/colearning_attacker/buffer_size", float(len(attacker_buffer)), global_step)
        if (episode + 1) % max(1, int(episodes) // 5) == 0:
            print(
                f"COLEARNING_DEFENDER episode={episode + 1}/{episodes} "
                f"def_buffer={len(defender_buffer)} atk_buffer={len(attacker_buffer)}"
            )
    return attacker


def _log_train_episode(
    writer: SummaryWriter,
    *,
    prefix: str,
    traj,
    losses: list[dict[str, float]],
    buffer_size: int,
    global_step: int,
) -> None:
    rewards_d = [float(t.defender_reward) for t in traj.transitions]
    rewards_a = [float(t.attacker_reward) for t in traj.transitions]
    clean = [float(t.info.get("clean_acc", float("nan"))) for t in traj.transitions]
    alpha = [float(t.info["defense_decision"].norm_bound_alpha) for t in traj.transitions]
    beta = [float(t.info["defense_decision"].trimmed_mean_beta) for t in traj.transitions]
    post = []
    for t in traj.transitions:
        decision = t.info["defense_decision"]
        post.append(float(decision.neuroclip_epsilon if decision.neuroclip_epsilon is not None else decision.prun_mask_rate or 0.0))
    writer.add_scalar(f"{prefix}/mean_defender_reward", float(np.nanmean(rewards_d)), global_step)
    writer.add_scalar(f"{prefix}/mean_attacker_reward", float(np.nanmean(rewards_a)), global_step)
    writer.add_scalar(f"{prefix}/mean_clean_acc", float(np.nanmean(clean)), global_step)
    writer.add_scalar(f"{prefix}/attack_success", 1.0 - float(np.nanmean(clean)), global_step)
    writer.add_scalar(f"{prefix}/alpha", float(np.nanmean(alpha)), global_step)
    writer.add_scalar(f"{prefix}/beta", float(np.nanmean(beta)), global_step)
    writer.add_scalar(f"{prefix}/post", float(np.nanmean(post)), global_step)
    writer.add_scalar(f"{prefix}/buffer_size", float(buffer_size), global_step)
    _log_loss_means(writer, prefix, losses, global_step)


def _log_loss_means(writer: SummaryWriter, prefix: str, losses: list[dict[str, float]], global_step: int) -> None:
    if not losses:
        return
    for key in sorted({k for loss in losses for k in loss}):
        values = [float(loss[key]) for loss in losses if key in loss and np.isfinite(float(loss[key]))]
        if values:
            writer.add_scalar(f"{prefix}/{key}", float(np.mean(values)), global_step)


def _merge_trajectories(left, right):
    left.transitions.extend(right.transitions)
    return left


def _evaluate_defender(
    *,
    coordinator_factory: Callable[[], FLCoordinator],
    obs_dim: int,
    td3_cfg: TD3Config,
    defender_name: str,
    defender_policy,
    attacker: TD3Agent,
    defender_action_cfg: DefenderActionConfig,
    seeds: list[int],
    horizon: int,
    csv_writer: csv.DictWriter,
    tb_writer: SummaryWriter,
) -> None:
    del td3_cfg
    clean_by_step: dict[int, list[float]] = {}
    success_by_step: dict[int, list[float]] = {}
    defender_reward_by_step: dict[int, list[float]] = {}
    attacker_reward_by_step: dict[int, list[float]] = {}
    alpha_by_step: dict[int, list[float]] = {}
    beta_by_step: dict[int, list[float]] = {}
    post_by_step: dict[int, list[float]] = {}
    for seed in seeds:
        env = _build_env(coordinator_factory, attacker, horizon, defender_action_cfg)
        def_buffer = ReplayBuffer(max(horizon, 1), obs_dim, 3)
        atk_buffer = ReplayBuffer(max(horizon, 1), obs_dim, 3)
        collector = TrajectoryCollector(
            env=env,
            defender=defender_policy,
            attacker=attacker,
            defender_buffer=def_buffer,
            attacker_buffer=atk_buffer,
            exploration_noise=0.0,
            store_attacker=False,
        )
        traj = collector.collect(horizon, seed=int(seed))
        for step, transition in enumerate(traj.transitions):
            clean = float(transition.info.get("clean_acc", float("nan")))
            d_reward = float(transition.defender_reward)
            a_reward = float(transition.attacker_reward)
            decision = transition.info["defense_decision"]
            post = (
                decision.neuroclip_epsilon
                if decision.neuroclip_epsilon is not None
                else decision.prun_mask_rate or 0.0
            )
            attack_success = 1.0 - clean
            _append(clean_by_step, step, clean)
            _append(success_by_step, step, attack_success)
            _append(defender_reward_by_step, step, d_reward)
            _append(attacker_reward_by_step, step, a_reward)
            _append(alpha_by_step, step, float(decision.norm_bound_alpha))
            _append(beta_by_step, step, float(decision.trimmed_mean_beta))
            _append(post_by_step, step, float(post))
            csv_writer.writerow(
                {
                    "defender": defender_name,
                    "seed": seed,
                    "step": step,
                    "clean_acc": clean,
                    "attack_success": attack_success,
                    "defender_reward": d_reward,
                    "attacker_reward": a_reward,
                    "alpha": float(decision.norm_bound_alpha),
                    "beta": float(decision.trimmed_mean_beta),
                    "post": float(post),
                }
            )
    for step in sorted(clean_by_step):
        tb_writer.add_scalar("eval/clean_acc", float(np.nanmean(clean_by_step[step])), step)
        tb_writer.add_scalar("eval/attack_success", float(np.nanmean(success_by_step[step])), step)
        tb_writer.add_scalar("eval/defender_reward", float(np.nanmean(defender_reward_by_step[step])), step)
        tb_writer.add_scalar("eval/attacker_reward", float(np.nanmean(attacker_reward_by_step[step])), step)
        tb_writer.add_scalar("defender/alpha", float(np.nanmean(alpha_by_step[step])), step)
        tb_writer.add_scalar("defender/beta", float(np.nanmean(beta_by_step[step])), step)
        tb_writer.add_scalar("defender/post", float(np.nanmean(post_by_step[step])), step)
    final_clean = float(np.nanmean(clean_by_step[max(clean_by_step)]))
    print(defender_name, "final_clean_acc=", round(final_clean, 4))


def _append(bucket: dict[int, list[float]], step: int, value: float) -> None:
    bucket.setdefault(step, []).append(float(value))


if __name__ == "__main__":
    main()
