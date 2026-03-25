"""
Meta-Stackelberg Learning on Federated Learning (MNIST).

Trains a meta-defense policy against multiple attack types using
the nested two-timescale Reptile meta-SG algorithm.

Usage:
    python train_meta_fl.py
    python train_meta_fl.py --N_D 50 --H 10 --K 2 --N_A 3
    tensorboard --logdir runs/meta_sg_fl
"""

import argparse
import torch
import numpy as np

from src.envs.fl_meta_env import FLMetaEnv
from src.agents.meta_agents import ContinuousPolicyAgent, FixedPolicyAgent
from src.algos.meta_sg import MetaSGTrainer, AttackerDomain


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-SG on FL (MNIST)")

    # FL environment
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_untargeted_attackers", type=int, default=2)
    parser.add_argument("--num_backdoor_attackers", type=int, default=1)
    parser.add_argument("--subsample_rate", type=float, default=1.0)
    parser.add_argument("--fl_lr", type=float, default=0.05)
    parser.add_argument("--post_defense", type=str, default="neuroclip",
                        choices=["neuroclip", "pruning"])

    # Meta-learning
    parser.add_argument("--N_D", type=int, default=20, help="Outer loop iterations")
    parser.add_argument("--K", type=int, default=2, help="Attack types sampled per iter")
    parser.add_argument("--eta", type=float, default=0.01, help="Adaptation step size")
    parser.add_argument("--kappa_D", type=float, default=0.001, help="Meta step size")

    # Inner loop
    parser.add_argument("--N_A", type=int, default=3, help="Attacker inner loop steps")
    parser.add_argument("--kappa_A", type=float, default=0.001, help="Attacker step size")

    # Trajectory
    parser.add_argument("--H", type=int, default=5, help="FL rounds per trajectory")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # Policy gradient
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Misc
    parser.add_argument("--scheme", type=str, default="reptile")
    parser.add_argument("--log_dir", type=str, default="runs/meta_sg_fl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/meta_sg_fl")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Attack domain
    parser.add_argument("--attacks", type=str, nargs="+",
                        default=["IPM", "BFL"],
                        help="Attack types to include in domain")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Create FL environment
    env_config = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "num_untargeted_attackers": args.num_untargeted_attackers,
        "num_backdoor_attackers": args.num_backdoor_attackers,
        "subsample_rate": args.subsample_rate,
        "fl_rounds": args.H,
        "lr": args.fl_lr,
        "post_defense": args.post_defense,
    }
    env = FLMetaEnv(env_config)

    obs_dim = env.state_dim
    act_dim = 3  # (alpha, beta, epsilon)

    print(f"FL Env: {args.dataset}, {args.num_clients} clients")
    print(f"State dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Post-defense: {args.post_defense}")
    print(f"Attacks: {args.attacks}")

    # 2. Create defender
    defender = ContinuousPolicyAgent(obs_dim, act_dim, hidden_dim=args.hidden_dim)
    print(f"Defender params: {sum(p.numel() for p in defender.parameters())}")

    # 3. Create attacker domain
    attacker_domain = AttackerDomain()
    adaptive_attacks = {"RL", "BRL"}

    for attack_type in args.attacks:
        if attack_type in adaptive_attacks:
            attacker = ContinuousPolicyAgent(obs_dim, act_dim,
                                             hidden_dim=args.hidden_dim)
            print(f"  {attack_type}: adaptive (ContinuousPolicyAgent)")
        else:
            attacker = FixedPolicyAgent(act_dim, name=attack_type)
            print(f"  {attack_type}: fixed (FixedPolicyAgent)")
        attacker_domain.register(attack_type, attacker)

    # 4. Create trainer
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
        defender_id="defender",
        attacker_id="attacker",
        config=config,
        obs_dim=obs_dim,
    )

    # 5. Train
    print(f"\nMeta-SG FL Training ({args.scheme})")
    print(f"  Outer iterations: {args.N_D}")
    print(f"  Types per iter: {args.K}")
    print(f"  Inner loop: {args.N_A} steps")
    print(f"  Trajectory: {args.H} FL rounds")
    print(f"  TensorBoard: tensorboard --logdir {args.log_dir}")
    print()

    trained_defender = trainer.train()

    # 6. Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    for attack_type in args.attacks:
        obs, infos = env.reset(options={"attack_type": attack_type})
        total_def_reward = 0
        for _ in range(args.H):
            obs_t = torch.tensor(obs["defender"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = trained_defender.get_action(obs_t)
            actions = {
                "defender": action.numpy(),
                "attacker": np.zeros(act_dim),
            }
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_def_reward += rewards["defender"]

        info = infos["defender"]
        print(f"  {attack_type:>4s}: acc={info['main_acc']:.3f} "
              f"bac={info['backdoor_acc']:.3f} "
              f"total_reward={total_def_reward:.3f}")

    print(f"\nCheckpoints: {args.save_dir}")
    print(f"TensorBoard: {args.log_dir}")


if __name__ == "__main__":
    main()
