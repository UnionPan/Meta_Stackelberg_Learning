
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import numpy as np

from src.utils.fl_utils import *
from src.utils.data_loader import get_datasets, poison_dataset, DatasetSplit
from src.models.cnn import ResNet18

def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is required if you want to use the environment with an agent expecting a single observation and action space
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    wrapper to convert from a ParallelEnv to an AEC env
    """
    env = FederatedLearningEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class FederatedLearningEnv(ParallelEnv):
    """
    A PettingZoo environment for simulating Federated Learning for the defender's perspective.
    """
    metadata = {'render.modes': ['human'], "name": "federated_learning_v0"}

    def __init__(self, num_clients=10, num_attackers=3, subsample_rate=1.0, dataset_name='mnist', lr=0.01, epochs=5, render_mode=None, attacker_types=['label_flip', 'backdoor', 'gaussian']):
        super(FederatedLearningEnv, self).__init__()
        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.subsample_rate = subsample_rate
        self.dataset_name = dataset_name
        self.model = ResNet18()
        self.lr = lr
        self.epochs = epochs
        self.render_mode = render_mode
        self.attacker_types = attacker_types
        self.attacker_type = self.attacker_types[0]
        self.flat_weights_len = len(self._flatten_weights(get_parameters(self.model)))

        self.possible_agents = ["defender", "attacker"]
        self.agents = self.possible_agents[:]

        # Define action and observation space
        self.action_spaces = {
            "defender": spaces.Discrete(10), # Placeholder
            "attacker": spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_weights_len,), dtype=np.float32)
        }
        
        self.observation_spaces = {
            "defender": spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_weights_len,), dtype=np.float32),
            "attacker": spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_weights_len,), dtype=np.float32)
        }

        self.train_dataset, self.test_dataset = get_datasets(self.dataset_name)
        
    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.agents = self.possible_agents[:]
        self.global_weights = get_parameters(self.model)
        
        # Sample an attacker type for this episode
        self.attacker_type = np.random.choice(self.attacker_types)

        # Create data splits for clients
        self.client_indices = self._split_data()

        # Poison data for attackers based on the attacker type
        self._poison_data(self.attacker_type)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        # 1. Defender chooses an action (aggregation rule)
        defender_action = actions["defender"]
        filter_rate = 0.1 # Placeholder

        # 2. Attacker chooses an action (malicious updates)
        attacker_action = self._get_attacker_action(self.attacker_type, actions["attacker"])

        # 3. Subsample clients
        num_subsampled_clients = int(self.num_clients * self.subsample_rate)
        subsampled_client_indices = np.random.choice(self.num_clients, num_subsampled_clients, replace=False)

        # 4. Clients train
        local_weights = []
        for client_idx in subsampled_client_indices:
            if client_idx in self.attacker_indices:
                # This is a malicious client, their action is determined by the attacker agent
                malicious_weights = self._unflatten_weights(attacker_action, self.global_weights)
                local_weights.append(malicious_weights)
            else:
                # This is a benign client
                client_model = copy.deepcopy(self.model)
                set_parameters(client_model, self.global_weights)
                
                train_loader = torch.utils.data.DataLoader(
                    DatasetSplit(self.train_dataset, list(self.client_indices[client_idx])),
                    batch_size=64,
                    shuffle=True
                )
                train_iter = iter(train_loader)

                train(client_model, train_iter, self.epochs, self.lr)
                local_weights.append(get_parameters(client_model))


        # 6. Aggregate weights
        self.global_weights = Clipped_Mean(self.global_weights, local_weights, max_norm=10, filter_rate=filter_rate)
        set_parameters(self.model, self.global_weights)

        # 7. Evaluate the model
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        loss, accuracy = test(self.model, test_loader)

        # 8. Calculate reward
        rewards = {
            "defender": accuracy,
            "attacker": -accuracy # Placeholder
        }

        # 9. Get observation
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def _get_obs(self, agent):
        return self._flatten_weights(self.global_weights)

    def _get_attacker_action(self, attacker_type, attacker_action):
        # This is a placeholder for the attacker's logic.
        # Based on the attacker_type, we will generate the malicious updates.
        if attacker_type == 'label_flip':
            # For label flipping, the attacker's action is the malicious weights.
            return attacker_action
        elif attacker_type == 'backdoor':
            # For backdoor attacks, the attacker's action is the malicious weights.
            return attacker_action
        elif attacker_type == 'gaussian':
            # For gaussian noise attack, the attacker adds noise to the global weights.
            return self._flatten_weights(self.global_weights) + attacker_action
        else:
            return attacker_action

    def _split_data(self):
        # Split training data among clients
        num_items = int(len(self.train_dataset) / self.num_clients)
        all_idxs = list(range(len(self.train_dataset)))
        client_indices = []
        for i in range(self.num_clients):
            client_indices.append(set(np.random.choice(all_idxs, num_items, replace=False)))
            all_idxs = list(set(all_idxs) - client_indices[i])
        return client_indices

    def _poison_data(self, attacker_type):
        # Poison the data of the attacker clients based on the attacker type
        # This is a placeholder, we will implement the different attack strategies here
        self.attacker_indices = np.random.choice(self.num_clients, self.num_attackers, replace=False)
        for i in self.attacker_indices:
            if attacker_type == 'label_flip':
                poison_dataset(self.train_dataset, self.dataset_name, base_class=5, target_class=7, poison_frac=1.0, pattern_type='square', data_idxs=list(self.client_indices[i]), poison_all=True)
            elif attacker_type == 'backdoor':
                poison_dataset(self.train_dataset, self.dataset_name, base_class=5, target_class=7, poison_frac=1.0, pattern_type='square', data_idxs=list(self.client_indices[i]), poison_all=True)
            elif attacker_type == 'gaussian':
                # For gaussian attack, we don't poison the data, but the attacker will add noise to the weights
                pass

    def _flatten_weights(self, weights):
        return np.concatenate([w.flatten() for w in weights])

    def _unflatten_weights(self, flat_weights, model_weights):
        new_weights = []
        start = 0
        for w in model_weights:
            end = start + w.size
            new_weights.append(flat_weights[start:end].reshape(w.shape))
            start = end
        return new_weights
