import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np

from .meta_env import MetaEnv

class OneHotObsWrapper(wrappers.BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_spaces = {
            agent: Box(0, 1, (self.env.observation_space(agent).n,), np.float32)
            for agent in self.env.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def _one_hot_obs(self, obs):
        new_obs = {}
        for agent, agent_obs in obs.items():
            one_hot = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
            one_hot[agent_obs] = 1
            new_obs[agent] = one_hot
        return new_obs

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed, options)
        return self._one_hot_obs(obs), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._one_hot_obs(obs), rewards, terminations, truncations, infos

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    env = MetaRPS(render_mode=render_mode)
    env = OneHotObsWrapper(env)
    return env

class MetaRPS(MetaEnv):
    metadata = {"render_modes": ["human"], "name": "meta_rps_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self._opponent_types = ['normal', 'aggressive', 'defensive']
        self.player_2_type = self._opponent_types[0]

        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(4) for agent in self.possible_agents}

        self.render_mode = render_mode

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @property
    def opponent_types(self):
        return self._opponent_types

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._last_actions = {agent: 3 for agent in self.possible_agents}
        self.player_2_type = np.random.choice(self.opponent_types)
        
        observations = {agent: self._last_actions[self.possible_agents[1 - i]] for i, agent in enumerate(self.possible_agents)}
        
        # Only player_2 has a specific type info
        infos = {agent: {} for agent in self.possible_agents}
        infos['player_2']['type'] = self.opponent_types.index(self.player_2_type) # Pass the index of the type

        return observations, infos

    def step(self, actions):
        reward1, reward2 = self._get_rewards(actions)
        rewards = {"player_1": reward1, "player_2": reward2}
        self._last_actions = actions
        observations = {agent: self._last_actions[self.possible_agents[1 - i]] for i, agent in enumerate(self.possible_agents)}
        terminations = {agent: True for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        print(f"Player 1 action: {self._last_actions['player_1']}, Player 2 action: {self._last_actions['player_2']}")
        print(f"Player 2 type: {self.player_2_type}")
        reward1, reward2 = self._get_rewards(self._last_actions)
        print(f"Player 1 reward: {reward1}, Player 2 reward: {reward2}")

    def _get_rewards(self, actions):
        action1 = actions["player_1"]
        action2 = actions["player_2"]
        if action1 == action2:
            reward1, reward2 = 0, 0
        elif (action1 - action2) % 3 == 1:
            reward1, reward2 = 1, -1
        else:
            reward1, reward2 = -1, 1
        if self.player_2_type == 'aggressive':
            if reward2 == 1:
                reward2 = 2
        elif self.player_2_type == 'defensive':
            if reward2 == -1:
                reward2 = -0.5
        return reward1, reward2

if __name__ == "__main__":
    env = raw_env(render_mode="human")
    observations, infos = env.reset()
    print("Initial observations:", observations)
    print("Initial infos:", infos)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("\nStep results:")
    print("Actions:", actions)
    print("Observations:", observations)
    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Infos:", infos)
