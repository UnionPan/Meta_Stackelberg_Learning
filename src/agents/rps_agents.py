import torch
import torch.nn as nn
import torch.nn.functional as F

class RPSAgent(nn.Module):
    """
    A base class for the Rock-Paper-Scissors agents with an actor and a critic.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(RPSAgent, self).__init__()
        self.body = nn.Linear(obs_space_size, 32)
        self.actor_head = nn.Linear(32, action_space_size)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.body(x))
        policy_logits = self.actor_head(x)
        value_estimate = self.critic_head(x)
        return F.log_softmax(policy_logits, dim=-1), value_estimate

class Player1(RPSAgent):
    """
    The policy for Player 1.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(Player1, self).__init__(obs_space_size, action_space_size)

class Player2(nn.Module):
    """
    The policy for Player 2, which takes its type as an additional input.
    """
    def __init__(self, obs_space_size, action_space_size, num_types, type_embedding_dim=4):
        super(Player2, self).__init__()
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        
        self.body = nn.Linear(obs_space_size + type_embedding_dim, 32)
        self.actor_head = nn.Linear(32, action_space_size)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, obs, type_idx):
        type_embedding = self.type_embedding(type_idx)
        x = torch.cat([obs, type_embedding], dim=-1)
        x = F.relu(self.body(x))
        policy_logits = self.actor_head(x)
        value_estimate = self.critic_head(x)
        return F.log_softmax(policy_logits, dim=-1), value_estimate

if __name__ == "__main__":
    # --- Test Player 1 ---
    print("--- Testing Player 1 ---")
    obs_size_p1 = 4
    action_size_p1 = 3
    player1 = Player1(obs_size_p1, action_size_p1)

    dummy_obs_p1 = torch.zeros(1, obs_size_p1)
    dummy_obs_p1[0, 1] = 1

    log_probs_p1, value_p1 = player1(dummy_obs_p1)
    print("Player 1 input (dummy observation):", dummy_obs_p1)
    print("Player 1 output (log probabilities):", log_probs_p1)
    print("Player 1 output (value estimate):", value_p1)
    print("\n" + "="*30 + "\n")


    # --- Test Player 2 ---
    print("--- Testing Player 2 ---")
    obs_size_p2 = 4
    action_size_p2 = 3
    num_types_p2 = 3
    player2 = Player2(obs_size_p2, action_size_p2, num_types_p2)

    dummy_obs_p2 = torch.zeros(1, obs_size_p2)
    dummy_obs_p2[0, 2] = 1
    dummy_type_p2 = torch.tensor([0])

    log_probs_p2, value_p2 = player2(dummy_obs_p2, dummy_type_p2)
    print("Player 2 input (dummy observation):", dummy_obs_p2)
    print("Player 2 input (dummy type):", dummy_type_p2)
    print("Player 2 output (log probabilities):", log_probs_p2)
    print("Player 2 output (value estimate):", value_p2)