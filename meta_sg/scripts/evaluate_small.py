"""Small Meta-SG train/eval run with TensorBoard logging.

Run:
    conda run -n gym_env python meta_sg/scripts/evaluate_small.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from torch.utils.tensorboard import SummaryWriter

from meta_sg.games.observations import obs_dim_for
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.evaluation import PolicyEvaluator, assess_convergence
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.types import ATTACK_DOMAIN


def main() -> None:
    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "meta_sg_small_eval", run_name)
    writer = SummaryWriter(log_dir)

    coordinator_factory = lambda: StubCoordinator(num_clients=8, num_attackers=2, seed=11)
    obs_dim = obs_dim_for(coordinator_factory().spec.empty_weights())
    attack_domain = [ATTACK_DOMAIN["ipm"], ATTACK_DOMAIN["rl"]]

    meta_cfg = MetaSGConfig(
        T=40,
        K=2,
        H_mnist=8,
        l=3,
        N_A=3,
        post_br_defender_updates=1,
        eval_every=4,
        dataset="mnist",
    )
    td3_cfg = TD3Config(
        hidden_dim=64,
        batch_size=16,
        buffer_capacity=5000,
        warmup_steps=0,
        exploration_noise=0.15,
    )

    trainer = MetaSGTrainer(
        coordinator_factory=coordinator_factory,
        attack_domain=attack_domain,
        meta_config=meta_cfg,
        td3_config=td3_cfg,
        obs_dim=obs_dim,
        log_interval=5,
        writer=writer,
    )
    result = trainer.train()

    conv = assess_convergence(result.defender_rewards, window=10)
    writer.add_scalar("convergence/converged", float(conv.converged), meta_cfg.T)
    writer.add_scalar("convergence/rolling_mean", conv.rolling_mean, meta_cfg.T)
    writer.add_scalar("convergence/rolling_std", conv.rolling_std, meta_cfg.T)
    writer.add_scalar("convergence/slope", conv.slope, meta_cfg.T)

    evaluator = PolicyEvaluator(
        coordinator_factory=coordinator_factory,
        horizon=12,
        obs_dim=obs_dim,
        eval_every=4,
    )
    seeds = (101, 102, 103)
    summaries = [
        evaluator.evaluate(
            "meta_defender",
            trainer.defender,
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=seeds,
        ),
        evaluator.evaluate(
            "constant_zero",
            ConstantActionPolicy([0.0, 0.0, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=seeds,
        ),
        evaluator.evaluate(
            "constant_strong_clip",
            ConstantActionPolicy([-0.8, 0.2, 0.0]),
            attack_domain,
            attacker_agents=trainer.attacker_agents,
            seeds=seeds,
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
