import torch

from src.envs.meta_rps import raw_env as rps_env
from src.agents.rps_agents import Player1, Player2
from src.algos.mappo import MAPPO
from src.utils.rps_parser import get_rps_args

def main(args):
    # 1. Initialize environment
    env = rps_env()
    env.reset()

    # 2. Initialize agents
    
    obs_space_p1 = env.observation_space('player_1').shape[0]
    action_space_p1 = env.action_space('player_1').n
    player1 = Player1(obs_space_p1, action_space_p1)

    obs_space_p2 = env.observation_space('player_2').shape[0]
    action_space_p2 = env.action_space('player_2').n
    num_types_p2 = len(env.opponent_types)
    player2 = Player2(obs_space_p2, action_space_p2, num_types_p2)

    agents = {'player_1': player1, 'player_2': player2}

    # 3. Initialize MAPPO trainer
    mappo = MAPPO(env, agents, 
                  lr=args.lr, 
                  gamma=args.gamma, 
                  gae_lambda=args.gae_lambda, 
                  clip_epsilon=args.clip_epsilon, 
                  value_loss_coef=args.value_loss_coef, 
                  entropy_coef=args.entropy_coef, 
                  rollout_length=args.rollout_length)

    # 4. Run training
    mappo.train(args.num_training_iterations)

if __name__ == "__main__":
    args = get_rps_args()
    main(args)