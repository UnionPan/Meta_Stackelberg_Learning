"""
Meta-Stackelberg Learning on Meta Rock-Paper-Scissors.

This is a testbed for the meta-SG algorithm before applying it to FL.
Player 1 (defender/leader) learns a meta-policy via Reptile.
Player 2 (attacker/follower) has 3 types: normal, aggressive, defensive.
Each type has its own PolicyAgent that learns a best-response in the inner loop.

Usage:
    python train_meta_rps.py
    python train_meta_rps.py --N_D 500 --H 128 --K 3
    tensorboard --logdir runs/meta_sg_rps
"""

import argparse
import torch
import numpy as np
import os

from src.envs.meta_rps import raw_env as rps_env
from src.agents.meta_agents import PolicyAgent
from src.algos.meta_sg import MetaSGTrainer, AttackerDomain
from src.utils.viz import (
    plot_all_policies_on_simplex,
    compute_rps_exploitability,
    get_policy_probs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-SG on Meta-RPS")

    # Meta-learning
    parser.add_argument("--N_D", type=int, default=300, help="Outer loop iterations")
    parser.add_argument("--K", type=int, default=3, help="Attack types sampled per iter")
    parser.add_argument("--eta", type=float, default=0.05, help="Adaptation step size")
    parser.add_argument("--kappa_D", type=float, default=0.01, help="Meta step size")

    # Inner loop
    parser.add_argument("--N_A", type=int, default=5, help="Attacker inner loop steps")
    parser.add_argument("--kappa_A", type=float, default=0.01, help="Attacker step size")

    # Trajectory
    parser.add_argument("--H", type=int, default=128, help="Trajectory horizon")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")

    # Policy gradient
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="Entropy bonus")
    parser.add_argument("--value_loss_coef", type=float, default=0.5)

    # Misc
    parser.add_argument("--scheme", type=str, default="reptile",
                        choices=["reptile", "debiased"])
    parser.add_argument("--log_dir", type=str, default="runs/meta_sg_rps")
    parser.add_argument("--save_dir", type=str, default="checkpoints/meta_sg_rps")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--no_plot", action="store_true", help="Skip simplex plot")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Create environment
    env = rps_env()
    obs_dim = env.observation_space("player_1").shape[0]
    act_dim = env.action_space("player_1").n

    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"Opponent types: {env.opponent_types}")

    # 2. Create defender (player_1) - the meta-learner
    defender = PolicyAgent(obs_dim, act_dim, hidden_dim=args.hidden_dim)

    # 3. Create attacker domain - one PolicyAgent per type
    attacker_domain = AttackerDomain()
    for type_name in env.opponent_types:
        attacker = PolicyAgent(obs_dim, act_dim, hidden_dim=args.hidden_dim)
        attacker_domain.register(type_name, attacker)
        print(f"  Registered attacker type: {type_name}")

    # 4. Create trainer with exploitability tracking
    config = {
        "N_D": args.N_D,
        "K": args.K,
        "eta": args.eta,
        "kappa_D": args.kappa_D,
        "N_A": args.N_A,
        "kappa_A": args.kappa_A,
        "H": args.H,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "scheme": args.scheme,
        "log_dir": args.log_dir,
        "save_dir": args.save_dir,
        "save_interval": args.save_interval,
    }

    trainer = MetaSGTrainer(
        env=env,
        defender=defender,
        attacker_domain=attacker_domain,
        defender_id="player_1",
        attacker_id="player_2",
        config=config,
        exploitability_fn=compute_rps_exploitability,
        obs_dim=obs_dim,
    )

    # 5. Train
    print(f"\nStarting meta-SG training ({args.scheme} scheme)")
    print(f"  Outer iterations: {args.N_D}")
    print(f"  Types per iter: {args.K}")
    print(f"  Inner loop steps: {args.N_A}")
    print(f"  Trajectory horizon: {args.H}")
    print(f"  TensorBoard: tensorboard --logdir {args.log_dir}")
    print()

    trained_defender = trainer.train()

    # 6. Evaluate final policies
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    obs_labels = ["Rock", "Paper", "Scissors", "Start"]

    print("\nDefender policy:")
    for i in range(obs_dim):
        probs = get_policy_probs(trained_defender, obs_dim, obs_index=i)
        exploit = compute_rps_exploitability(probs)
        print(
            f"  Obs={obs_labels[i]:>8s} -> "
            f"P(R)={probs[0]:.3f} P(P)={probs[1]:.3f} P(S)={probs[2]:.3f} "
            f"exploit={exploit:.4f}"
        )

    print("\nAttacker policies:")
    for type_name in attacker_domain:
        attacker = attacker_domain.get(type_name)
        if isinstance(attacker, PolicyAgent):
            probs = get_policy_probs(attacker, obs_dim, obs_index=-1)
            print(
                f"  [{type_name:>10s}] Start -> "
                f"P(R)={probs[0]:.3f} P(P)={probs[1]:.3f} P(S)={probs[2]:.3f}"
            )

    # 7. Plot policy evolution on simplex
    if not args.no_plot:
        histories = trainer.get_policy_histories()

        defender_hist = histories.get("defender", [])
        attacker_hists = {}
        for type_name in attacker_domain:
            key = f"attacker_{type_name}"
            if key in histories and len(histories[key]) > 0:
                attacker_hists[type_name] = histories[key]

        if defender_hist:
            plot_dir = os.path.join(args.log_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            save_path = os.path.join(plot_dir, "simplex_evolution.png")
            fig = plot_all_policies_on_simplex(
                defender_hist,
                attacker_hists,
                title="Meta-SG Policy Evolution (RPS)",
                labels=("Rock", "Paper", "Scissors"),
                save_path=save_path,
            )
            print(f"\nSimplex plot saved to: {save_path}")

            # Log final simplex figure to TensorBoard
            from src.utils.viz import fig_to_tensor
            trainer.writer = torch.utils.tensorboard.SummaryWriter(args.log_dir)
            img_tensor = fig_to_tensor(fig)
            trainer.writer.add_image("simplex/final_evolution", img_tensor, args.N_D)
            trainer.writer.close()

    print(f"\nCheckpoints saved to: {args.save_dir}")
    print(f"TensorBoard logs: {args.log_dir}")


if __name__ == "__main__":
    main()
