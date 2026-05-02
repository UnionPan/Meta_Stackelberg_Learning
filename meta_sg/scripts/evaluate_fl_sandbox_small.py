"""Small real-FL Meta-SG run through fl_sandbox.

This is intentionally tiny. It validates the end-to-end real FL adapter and
produces TensorBoard diagnostics; it is not a paper-scale result.

Run:
    conda run -n gym_env python meta_sg/scripts/evaluate_fl_sandbox_small.py
"""
from __future__ import annotations

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


def main() -> None:
    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "meta_sg_fl_sandbox_small", run_name)
    writer = SummaryWriter(log_dir)

    sandbox_config = SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device="cpu",
        num_clients=2,
        num_attackers=1,
        subsample_rate=1.0,
        local_epochs=1,
        batch_size=16,
        eval_batch_size=128,
        fltrust_root_size=0,
        max_client_samples_per_client=16,
        max_eval_samples=128,
    )
    coordinator_factory = lambda: FLSandboxCoordinatorAdapter(sandbox_config)
    obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights())

    meta_cfg = MetaSGConfig(
        T=8,
        K=1,
        H_mnist=3,
        l=1,
        N_A=1,
        post_br_defender_updates=0,
        eval_every=1,
        dataset="mnist",
    )
    td3_cfg = TD3Config(
        hidden_dim=32,
        batch_size=4,
        buffer_capacity=500,
        warmup_steps=0,
        exploration_noise=0.1,
    )

    trainer = MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=[ATTACK_DOMAIN["ipm"]],
        meta_config=meta_cfg,
        td3_config=td3_cfg,
        obs_dim=obs_dim,
        log_interval=1,
        writer=writer,
    )
    result = trainer.train()
    conv = assess_convergence(result.defender_rewards, window=4, min_points=8, std_tol=0.08, slope_tol=0.04)
    writer.add_scalar("convergence/converged", float(conv.converged), meta_cfg.T)
    writer.add_scalar("convergence/rolling_mean", conv.rolling_mean, meta_cfg.T)
    writer.add_scalar("convergence/rolling_std", conv.rolling_std, meta_cfg.T)
    writer.add_scalar("convergence/slope", conv.slope, meta_cfg.T)

    evaluator = PolicyEvaluator(
        coordinator_factory=coordinator_factory,
        horizon=3,
        obs_dim=obs_dim,
        eval_every=1,
    )
    summaries = [
        evaluator.evaluate(
            "meta_defender",
            trainer.defender,
            [ATTACK_DOMAIN["ipm"]],
            attacker_agents=trainer.attacker_agents,
            seeds=(101, 102),
        ),
        evaluator.evaluate(
            "constant_zero",
            ConstantActionPolicy([0.0, 0.0, 0.0]),
            [ATTACK_DOMAIN["ipm"]],
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

    writer.flush()
    writer.close()

    print("LOG_DIR", log_dir)
    print("TRAIN_REWARDS", result.defender_rewards)
    print("CONVERGENCE", conv)
    for summary in summaries:
        print(
            "EVAL",
            summary.name,
            "mean_reward=", round(summary.mean_reward, 4),
            "std=", round(summary.std_reward, 4),
            "worst=", round(summary.worst_reward, 4),
            "clean=", round(summary.mean_final_clean_acc, 4),
            "backdoor=", round(summary.mean_final_backdoor_acc, 4),
        )


if __name__ == "__main__":
    main()
