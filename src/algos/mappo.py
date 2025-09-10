import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class MAPPO:
    """
    A generic Multi-Agent PPO implementation with TensorBoard logging.
    """
    def __init__(self, env, agents, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01, rollout_length=2048, log_dir="runs/mappo"):
        self.env = env
        self.agents = agents
        self.agent_ids = self.env.possible_agents

        self.optimizers = {agent_id: torch.optim.Adam(self.agents[agent_id].parameters(), lr=lr) for agent_id in self.agent_ids}

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rollout_length = rollout_length

        self.writer = SummaryWriter(log_dir)

    def train(self, num_training_iterations):
        global_step = 0
        for i in range(num_training_iterations):
            # 1. Collect trajectories
            trajectories = self._collect_trajectories()

            # 2. Compute advantages
            advantages = {}
            returns = {}
            for agent_id in self.agent_ids:
                advantages[agent_id], returns[agent_id] = self._compute_advantages(trajectories, agent_id)

            # 3. Update policies
            self._update_policies(trajectories, advantages, returns, global_step)
            
            global_step += self.rollout_length
            print(f"Iteration {i+1}/{num_training_iterations}, Total Steps: {global_step}")

    def _collect_trajectories(self):
        trajectories = {agent_id: {'obs': [], 'actions': [], 'rewards': [], 'log_probs': [], 'values': [], 'dones': []} for agent_id in self.agent_ids}

        obs, infos = self.env.reset()
        done = False

        for _ in range(self.rollout_length):
            if done:
                obs, infos = self.env.reset()

            actions = {}
            log_probs = {}
            values = {}
            with torch.no_grad():
                for agent_id in self.agent_ids:
                    obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0) # Optimized tensor creation
                    
                    # Check if the agent is Player2 and needs type_idx
                    if agent_id == 'player_2': # Check if it's player_2
                        type_tensor = torch.tensor([self.env.opponent_types.index(self.env.player_2_type)], dtype=torch.long) # Get type from env
                        log_prob, value = self.agents[agent_id](obs_tensor, type_tensor)
                    else:
                        log_prob, value = self.agents[agent_id](obs_tensor)

                    action = torch.distributions.Categorical(logits=log_prob).sample()
                    actions[agent_id] = action.item()
                    log_probs[agent_id] = log_prob[0, action.item()]
                    values[agent_id] = value

            next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions) # Capture next_infos

            for agent_id in self.agent_ids:
                trajectories[agent_id]['obs'].append(obs_tensor)
                trajectories[agent_id]['actions'].append(action)
                trajectories[agent_id]['rewards'].append(rewards[agent_id])
                trajectories[agent_id]['log_probs'].append(log_probs[agent_id])
                trajectories[agent_id]['values'].append(values[agent_id])
                trajectories[agent_id]['dones'].append(terminations[agent_id])

            obs = next_obs
            infos = next_infos # Update infos for the next step
            done = any(terminations.values())

        # Get the last value for bootstrapping
        with torch.no_grad():
            for agent_id in self.agent_ids:
                obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0) # Optimized tensor creation
                if agent_id == 'player_2': # Check if it's player_2
                    type_tensor = torch.tensor([self.env.opponent_types.index(self.env.player_2_type)], dtype=torch.long) # Get type from env
                    _, last_value = self.agents[agent_id](obs_tensor, type_tensor)
                else:
                    _, last_value = self.agents[agent_id](obs_tensor)
                trajectories[agent_id]['values'].append(last_value)

        return trajectories

    def _compute_advantages(self, trajectories, agent_id):
        rewards = trajectories[agent_id]['rewards']
        values = trajectories[agent_id]['values']
        dones = trajectories[agent_id]['dones']
        
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t+1]
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages.insert(0, last_advantage)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return advantages, returns

    def _update_policies(self, trajectories, advantages, returns, global_step):
        for agent_id in self.agent_ids:
            self._update_agent_policy(agent_id, trajectories[agent_id], advantages[agent_id], returns[agent_id], global_step)

    def _update_agent_policy(self, agent_id, trajectory, advantages, returns, global_step):
        agent = self.agents[agent_id]
        optimizer = self.optimizers[agent_id]

        obs = torch.cat(trajectory['obs'])
        actions = torch.cat(trajectory['actions'])
        old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float32)

        # Get new log_probs, values, and entropy from the policy
        # Check if the agent is Player2 and needs type_idx
        if agent_id == 'player_2':
            type_tensor = torch.full((len(obs),), self.env.opponent_types.index(self.env.player_2_type), dtype=torch.long)
            policy_logits, new_values = agent(obs, type_tensor)
        else:
            policy_logits, new_values = agent(obs)
        
        # Create a Categorical distribution from the log probabilities tensor
        new_log_probs_dist = torch.distributions.Categorical(logits=policy_logits)

        new_log_probs = new_log_probs_dist.log_prob(actions)
        entropy = new_log_probs_dist.entropy().mean()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values.squeeze(), returns)

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.writer.add_scalar(f'loss/policy_loss_{agent_id}', policy_loss.item(), global_step)
        self.writer.add_scalar(f'loss/value_loss_{agent_id}', value_loss.item(), global_step)
        self.writer.add_scalar(f'loss/total_loss_{agent_id}', loss.item(), global_step)
        self.writer.add_scalar(f'policy/entropy_{agent_id}', entropy.item(), global_step)