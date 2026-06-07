"""Run full Meta-SG co-learning and write TensorBoard diagnostics."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.evaluation import PolicyEvaluator, assess_convergence
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, SandboxConfig
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.types import ATTACK_DOMAIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("stub", "fl_sandbox"), default="stub")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-root", default="runs/meta_sg_colearning_full")
    parser.add_argument("--outer-iters", type=int, default=30)
    parser.add_argument("--tasks-per-iter", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--defender-updates", type=int, default=3)
    parser.add_argument("--attacker-updates", type=int, default=3)
    parser.add_argument("--post-br-defender-updates", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=5000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--num-attackers", type=int, default=2)
    parser.add_argument("--client-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--batch-size-fl", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def total_colearning_training_rounds(
    *,
    outer_iters: int,
    tasks_per_iter: int,
    horizon: int,
    post_br_defender_updates: int,
) -> int:
    """FL rounds collected by adaptive co-learning train rollouts."""
    rollouts_per_task = 1 + (1 if post_br_defender_updates > 0 else 0)
    return int(outer_iters) * int(tasks_per_iter) * int(horizon) * rollouts_per_task


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_root) / f"{args.backend}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    coordinator_factory = _coordinator_factory(args)
    obs_dim = obs_dim_for(
        coordinator_factory().spec.empty_weights(),
        history_len=int(args.history_len),
    )
    total_rounds = total_colearning_training_rounds(
        outer_iters=args.outer_iters,
        tasks_per_iter=args.tasks_per_iter,
        horizon=args.horizon,
        post_br_defender_updates=args.post_br_defender_updates,
    )

    writer = SummaryWriter(str(log_dir))
    _write_config_scalars(writer, args, total_rounds)

    meta_cfg = MetaSGConfig(
        T=int(args.outer_iters),
        K=int(args.tasks_per_iter),
        H_mnist=int(args.horizon),
        l=int(args.defender_updates),
        N_A=int(args.attacker_updates),
        post_br_defender_updates=int(args.post_br_defender_updates),
        eval_every=int(args.eval_every),
        warmup_steps=0,
        history_len=int(args.history_len),
        dataset="mnist",
    )
    td3_cfg = TD3Config(
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        buffer_capacity=int(args.buffer_capacity),
        warmup_steps=0,
        exploration_noise=float(args.exploration_noise),
    )

    print("LOG_DIR", log_dir)
    print("TOTAL_TRAINING_FL_ROUNDS", total_rounds)
    print(
        "COLEARNING",
        "defender=TD3(alpha,beta,post_param)+Reptile",
        "attacker=TD3 best-response",
        "attack_domain=rl",
    )

    trainer = MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=[ATTACK_DOMAIN["rl"]],
        meta_config=meta_cfg,
        td3_config=td3_cfg,
        obs_dim=obs_dim,
        log_interval=max(1, min(5, int(args.outer_iters))),
        writer=writer,
    )
    result = trainer.train()

    conv = assess_convergence(
        result.defender_rewards,
        window=max(3, min(10, int(args.outer_iters) // 2 or 3)),
        min_points=min(8, int(args.outer_iters)),
        std_tol=0.08,
        slope_tol=0.04,
    )
    writer.add_scalar("convergence/converged", float(conv.converged), meta_cfg.T)
    writer.add_scalar("convergence/rolling_mean", conv.rolling_mean, meta_cfg.T)
    writer.add_scalar("convergence/rolling_std", conv.rolling_std, meta_cfg.T)
    writer.add_scalar("convergence/slope", conv.slope, meta_cfg.T)

    evaluator = PolicyEvaluator(
        coordinator_factory=coordinator_factory,
        horizon=max(2, min(int(args.horizon), 8)),
        obs_dim=obs_dim,
        eval_every=int(args.eval_every),
        history_len=int(args.history_len),
    )
    summaries = [
        evaluator.evaluate(
            "meta_defender_vs_learned_attacker",
            trainer.defender,
            [ATTACK_DOMAIN["rl"]],
            attacker_agents=trainer.attacker_agents,
            seeds=(101, 102),
        ),
        evaluator.evaluate(
            "constant_mid_vs_learned_attacker",
            ConstantActionPolicy([0.0, 0.0, 0.0]),
            [ATTACK_DOMAIN["rl"]],
            attacker_agents=trainer.attacker_agents,
            seeds=(101, 102),
        ),
    ]
    for summary in summaries:
        writer.add_scalar(f"eval/{summary.name}/mean_reward", summary.mean_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/std_reward", summary.std_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/worst_reward", summary.worst_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/mean_final_clean_acc", summary.mean_final_clean_acc, 0)
        writer.add_scalar(f"eval/{summary.name}/mean_final_backdoor_acc", summary.mean_final_backdoor_acc, 0)
        print(
            "EVAL",
            summary.name,
            "mean_reward=", round(summary.mean_reward, 4),
            "clean=", round(summary.mean_final_clean_acc, 4),
            "backdoor=", round(summary.mean_final_backdoor_acc, 4),
        )

    writer.flush()
    writer.close()
    print("CONVERGENCE", conv)


def _coordinator_factory(args: argparse.Namespace):
    if args.backend == "stub":
        return lambda: StubCoordinator(
            num_clients=int(args.num_clients),
            num_attackers=int(args.num_attackers),
            subsample_rate=1.0,
            seed=int(args.seed),
        )

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


def _write_config_scalars(writer: SummaryWriter, args: argparse.Namespace, total_rounds: int) -> None:
    writer.add_scalar("config/total_training_fl_rounds", float(total_rounds), 0)
    writer.add_scalar("config/outer_iters", float(args.outer_iters), 0)
    writer.add_scalar("config/tasks_per_iter", float(args.tasks_per_iter), 0)
    writer.add_scalar("config/horizon", float(args.horizon), 0)
    writer.add_scalar("config/defender_updates", float(args.defender_updates), 0)
    writer.add_scalar("config/attacker_updates", float(args.attacker_updates), 0)
    writer.add_scalar("config/post_br_defender_updates", float(args.post_br_defender_updates), 0)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    main()
