
import argparse

def get_rps_args():
    """
    Parses the arguments for the RPS experiment.
    """
    parser = argparse.ArgumentParser(description='RPS Experiment Arguments')

    # --- PPO Hyperparameters ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')

    # --- Training Parameters ---
    parser.add_argument('--rollout_length', type=int, default=2048, help='Rollout length')
    parser.add_argument('--num_training_iterations', type=int, default=100, help='Number of training iterations')

    # --- Environment Parameters ---
    parser.add_argument('--player2_type', type=str, default='normal', choices=['normal', 'aggressive', 'defensive'], help='The type of player 2')

    args = parser.parse_args()
    return args
