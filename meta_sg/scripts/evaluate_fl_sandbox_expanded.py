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
    parser.add_argument("--client-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--meta-step", type=float, default=0.10)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--attack-domain", default="ipm")
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
        device="cpu",
        seed=42,
        num_clients=4,
        num_attackers=1,
        subsample_rate=1.0,
        local_epochs=1,
        lr=0.05,
        batch_size=32,
        eval_batch_size=256,
        fltrust_root_size=0,
        max_client_samples_per_client=args.client_samples,
        max_eval_samples=args.eval_samples,
        parallel_clients=1,
    )
    coordinator_factory = lambda: FLSandboxCoordinatorAdapter(sandbox_config)
    obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights())

    meta_cfg = MetaSGConfig(
        T=args.T,
        K=args.K,
        H_mnist=args.H,
        l=args.l,
        N_A=1,
        post_br_defender_updates=0,
        eval_every=1,
        dataset="mnist",
        meta_update_step=args.meta_step,
        warmup_steps=0,
    )
    td3_cfg = TD3Config(
        hidden_dim=48,
        batch_size=8,
        buffer_capacity=1500,
        warmup_steps=0,
        exploration_noise=args.noise,
    )

    trainer = MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=attack_domain,
        meta_config=meta_cfg,
        td3_config=td3_cfg,
        obs_dim=obs_dim,
        log_interval=max(1, args.T // 12),
        writer=writer,
    )
    result = trainer.train()
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

    evaluator = PolicyEvaluator(
        coordinator_factory=coordinator_factory,
        horizon=args.H,
        obs_dim=obs_dim,
        eval_every=1,
    )
    summaries = [
        evaluator.evaluate(
            "meta_defender",
            trainer.defender,
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
        ),
        evaluator.evaluate_after_adaptation(
            "meta_defender_adapted",
            trainer.defender,
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
            adaptation_horizon=args.H,
            adaptation_updates=args.l,
            exploration_noise=args.noise,
        ),
        evaluator.evaluate(
            "mid_action",
            ConstantActionPolicy([0.0, 0.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
        ),
        evaluator.evaluate(
            "low_trim",
            ConstantActionPolicy([0.0, -1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
        ),
        evaluator.evaluate(
            "strong_trim",
            ConstantActionPolicy([0.0, 1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
        ),
        evaluator.evaluate(
            "strong_clip_trim",
            ConstantActionPolicy([-0.7, 1.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=(201, 202, 203),
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


if __name__ == "__main__":
    main()
