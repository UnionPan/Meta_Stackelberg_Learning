"""Expanded real-FL Meta-SG run through fl_sandbox.

This is a stability-oriented experiment: still far smaller than the paper,
but large enough to diagnose whether the meta defender reward curve settles.

Run:
    conda run -n gym_env python meta_sg/scripts/evaluate_fl_sandbox_expanded.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.tensorboard import SummaryWriter

from fl_sandbox.federation.runner import SandboxConfig
from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.evaluation import PolicyEvaluator, assess_convergence
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter
from meta_sg.strategies.types import ATTACK_DOMAIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=48)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-attackers", type=int, default=1)
    parser.add_argument("--subsample-rate", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--client-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--parallel-clients", type=int, default=1)
    parser.add_argument("--meta-step", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--buffer-capacity", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--attack-domain", default="ipm")
    parser.add_argument("--env-eval-every", type=int, default=1)
    parser.add_argument("--adaptation-horizon", type=int, default=None)
    parser.add_argument("--adaptation-episodes", type=int, default=1)
    parser.add_argument("--adaptation-updates", type=int, default=None)
    parser.add_argument("--adaptation-noise", type=float, default=None)
    parser.add_argument("--adaptation-lr-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-iteration", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-seeds", default="201,202,203")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.abspath(os.path.join("runs", "meta_sg_fl_sandbox_expanded", run_name))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    attack_domain = [
        ATTACK_DOMAIN[name.strip()]
        for name in args.attack_domain.split(",")
        if name.strip()
    ]
    if not attack_domain:
        raise ValueError("--attack-domain must contain at least one attack name")

    sandbox_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=args.device,
        seed=42,
        num_clients=args.num_clients,
        num_attackers=args.num_attackers,
        subsample_rate=args.subsample_rate,
        local_epochs=1,
        lr=0.05,
        batch_size=32,
        eval_batch_size=256,
        fltrust_root_size=0,
        max_client_samples_per_client=args.client_samples,
        max_eval_samples=args.eval_samples,
        parallel_clients=args.parallel_clients,
    )
    coordinator_factory = lambda: FLSandboxCoordinatorAdapter(sandbox_config)
    obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights(), history_len=args.history_len)

    resume_iteration = args.resume_iteration
    if resume_iteration is None:
        resume_iteration = _infer_resume_iteration(args.resume_dir)
    if resume_iteration < 0:
        raise ValueError("--resume-iteration must be >= 0")
    if resume_iteration > args.T:
        raise ValueError("--resume-iteration cannot be greater than --T")
    train_iterations = args.T - resume_iteration

    meta_cfg = MetaSGConfig(
        T=train_iterations,
        K=args.K,
        H_mnist=args.H,
        l=args.l,
        N_A=1,
        post_br_defender_updates=0,
        eval_every=args.env_eval_every,
        dataset="mnist",
        meta_update_step=args.meta_step,
        warmup_steps=args.warmup_steps,
        history_len=args.history_len,
    )
    td3_cfg = TD3Config(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        warmup_steps=0,
        exploration_noise=args.noise,
    )
    train_buffer_per_task = args.warmup_steps + args.H
    adapt_h = args.adaptation_horizon or args.H
    adapt_buffer = adapt_h * args.adaptation_episodes
    print(
        "EFFECTIVE_BUFFERS "
        f"train_per_task={train_buffer_per_task} "
        f"train_capacity={args.buffer_capacity} "
        f"adaptation={adapt_buffer} "
        f"adaptation_horizon={adapt_h} "
        f"adaptation_episodes={args.adaptation_episodes} "
        f"env_eval_every={args.env_eval_every}"
    )
    if args.buffer_capacity < train_buffer_per_task:
        print(
            "WARNING buffer_capacity is smaller than warmup_steps + H; "
            "older training transitions will be overwritten before TD3 updates."
        )

    trainer = MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=attack_domain,
        meta_config=meta_cfg,
        td3_config=td3_cfg,
        obs_dim=obs_dim,
        device=_resolve_torch_device(args.device),
        log_interval=args.log_interval or max(1, args.T // 12),
        writer=writer,
        checkpoint_dir=os.path.abspath(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_every,
        start_iteration=resume_iteration,
        total_iterations=args.T,
    )
    if args.resume_dir:
        trainer.load(os.path.abspath(args.resume_dir))
    result = trainer.train()
    if args.checkpoint_dir:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        trainer.save(checkpoint_dir)
        print("CHECKPOINT_DIR", checkpoint_dir)

    conv = assess_convergence(
        result.defender_rewards,
        window=min(10, max(4, args.T // 4)),
        min_points=max(12, args.T // 2),
        std_tol=0.08,
        slope_tol=0.015,
    )
    writer.add_scalar("convergence/converged", float(conv.converged), meta_cfg.T)
    writer.add_scalar("convergence/rolling_mean", conv.rolling_mean, meta_cfg.T)
    writer.add_scalar("convergence/rolling_std", conv.rolling_std, meta_cfg.T)
    writer.add_scalar("convergence/slope", conv.slope, meta_cfg.T)

    if args.skip_final_eval:
        writer.flush()
        writer.close()
        print("LOG_DIR", log_dir)
        print("TRAIN_REWARDS", [round(value, 4) for value in result.defender_rewards])
        print("CONVERGENCE", conv)
        return

    evaluator = PolicyEvaluator(
        coordinator_factory=coordinator_factory,
        horizon=args.H,
        obs_dim=obs_dim,
        eval_every=args.env_eval_every,
        history_len=args.history_len,
    )
    eval_seeds = tuple(
        int(seed.strip())
        for seed in args.eval_seeds.split(",")
        if seed.strip()
    )
    if not eval_seeds:
        raise ValueError("--eval-seeds must contain at least one integer seed")
    summaries = [
        evaluator.evaluate(
            "meta_defender",
            trainer.defender,
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
        ),
        evaluator.evaluate_after_adaptation(
            "meta_defender_adapted",
            trainer.defender,
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
            adaptation_horizon=args.adaptation_horizon or args.H,
            adaptation_episodes=args.adaptation_episodes,
            adaptation_updates=args.adaptation_updates or args.l,
            exploration_noise=args.adaptation_noise if args.adaptation_noise is not None else args.noise,
            adaptation_lr_scale=args.adaptation_lr_scale,
        ),
        evaluator.evaluate(
            "mid_action",
            ConstantActionPolicy([0.0, 0.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
        ),
        evaluator.evaluate(
            "low_trim",
            ConstantActionPolicy([0.0, -1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
        ),
        evaluator.evaluate(
            "strong_trim",
            ConstantActionPolicy([0.0, 1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
        ),
        evaluator.evaluate(
            "strong_clip_trim",
            ConstantActionPolicy([-0.7, 1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=eval_seeds,
        ),
    ]
    for summary in summaries:
        writer.add_scalar(f"eval/{summary.name}/mean_reward", summary.mean_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/std_reward", summary.std_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/worst_reward", summary.worst_reward, 0)
        writer.add_scalar(f"eval/{summary.name}/mean_final_clean_acc", summary.mean_final_clean_acc, 0)
        writer.add_scalar(f"eval/{summary.name}/mean_final_backdoor_acc", summary.mean_final_backdoor_acc, 0)

    writer.flush()
    writer.close()

    print("LOG_DIR", log_dir)
    print("TRAIN_REWARDS", [round(value, 4) for value in result.defender_rewards])
    print("CONVERGENCE", conv)
    for summary in summaries:
        print(
            "EVAL",
            summary.name,
            "mean_reward=",
            round(summary.mean_reward, 4),
            "std=",
            round(summary.std_reward, 4),
            "worst=",
            round(summary.worst_reward, 4),
            "clean=",
            round(summary.mean_final_clean_acc, 4),
            "backdoor=",
            round(summary.mean_final_backdoor_acc, 4),
        )


def _resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _infer_resume_iteration(resume_dir: str | None) -> int:
    if not resume_dir:
        return 0
    name = os.path.basename(os.path.normpath(resume_dir))
    prefix = "iter_"
    if name.startswith(prefix):
        try:
            return int(name[len(prefix):])
        except ValueError:
            pass
    return 0


if __name__ == "__main__":
    main()
