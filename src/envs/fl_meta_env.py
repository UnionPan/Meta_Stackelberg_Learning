"""
Federated Learning Meta-Environment for Meta-Stackelberg Learning.

A PettingZoo ParallelEnv where each step() executes one FL round.
The defender chooses aggregation hyperparameters (alpha, beta, epsilon).
The attacker's action parameterizes adaptive attacks (ignored for non-adaptive).

Designed to plug directly into MetaSGTrainer.
"""

import numpy as np
import copy
import random
import torch
from torch.utils.data import DataLoader

from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from ..models.cnn import MNISTClassifier, get_compressed_state
from ..utils.fl_utils import (
    get_parameters, set_parameters, train, test,
    weights_to_vector, vector_to_weights,
)
from ..utils.data_loader import (
    get_datasets, poison_dataset, add_pattern_bd, DatasetSplit,
)
from ..attacks import create_attack, AttackStrategy
from ..defenses import apply_post_defense

import more_itertools as mit


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FLMetaEnv(ParallelEnv):
    """
    Federated Learning environment for meta-Stackelberg games.

    Args:
        config: dict with keys:
            - dataset: "mnist" or "cifar10"
            - num_clients: number of FL clients (10 for pre-training, 100 for online)
            - num_untargeted_attackers: number of untargeted poisoning attackers
            - num_backdoor_attackers: number of backdoor attackers
            - subsample_rate: fraction of clients sampled per round
            - fl_rounds: episode length (H)
            - lr: client learning rate
            - batch_size: local training batch size
            - post_defense: "neuroclip" or "pruning"
            - base_class: source class for backdoor
            - target_class: target class for backdoor
            - pattern_type: backdoor pattern type
            - root_data_size: number of root data samples
            - seed: random seed for data splits
    """

    metadata = {"render_modes": ["human"], "name": "fl_meta_v0"}

    DEFAULT_CONFIG = {
        "dataset": "mnist",
        "num_clients": 10,
        "num_untargeted_attackers": 2,
        "num_backdoor_attackers": 1,
        "subsample_rate": 1.0,  # 1.0 for small pre-training env
        "fl_rounds": 200,
        "lr": 0.05,
        "batch_size": 128,
        "post_defense": "neuroclip",
        "base_class": 1,
        "target_class": 7,
        "pattern_type": "square",
        "root_data_size": 100,
        "seed": 42,
    }

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self.render_mode = render_mode
        self.possible_agents = ["defender", "attacker"]
        self.agents = self.possible_agents[:]

        # Build model and data
        self._setup_model()
        self._setup_data()
        self._setup_spaces()

        # Attack strategy (set via reset options)
        self._attack_strategy = None
        self._attack_type = None

        # Episode state
        self.round = 0
        self.initial_weights = get_parameters(self.model)

    def _setup_model(self):
        """Initialize the FL model."""
        if self.config["dataset"] == "mnist":
            self.model = MNISTClassifier()
        else:
            from ..models.cnn import ResNet18
            self.model = ResNet18()
        self.model.to(DEVICE)

    def _setup_data(self):
        """Load datasets, create client splits, prepare poisoned data."""
        cfg = self.config
        self.train_dataset, self.test_dataset = get_datasets(cfg["dataset"])

        # Create client data splits (iid)
        random.seed(cfg["seed"])
        num_items = len(self.train_dataset) // cfg["num_clients"]
        all_idxs = list(range(len(self.train_dataset)))
        self.client_data_idxs = []
        for _ in range(cfg["num_clients"]):
            selected = set(random.sample(all_idxs, min(num_items, len(all_idxs))))
            self.client_data_idxs.append(selected)
            all_idxs = list(set(all_idxs) - selected)

        # Assign attacker IDs
        random.seed(cfg["seed"] + 100)
        self.untargeted_att_ids = random.sample(
            range(cfg["num_clients"]), cfg["num_untargeted_attackers"]
        )
        random.seed(cfg["seed"] + 200)
        remaining = [i for i in range(cfg["num_clients"])
                     if i not in self.untargeted_att_ids]
        self.backdoor_att_ids = random.sample(
            remaining, min(cfg["num_backdoor_attackers"], len(remaining))
        )
        self.all_att_ids = self.untargeted_att_ids + self.backdoor_att_ids

        # Prepare poisoned datasets for backdoor attacks
        self._prepare_poisoned_data()

        # Prepare eval loaders
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=cfg["batch_size"], shuffle=False
        )

        # Root data loader (small subset for server-side evaluation)
        from torch.utils.data import RandomSampler
        root_sampler = RandomSampler(
            self.train_dataset, num_samples=cfg["root_data_size"], replacement=True
        )
        self.root_loader = DataLoader(
            self.train_dataset, batch_size=cfg["root_data_size"],
            sampler=root_sampler
        )
        self.root_iter = mit.seekable(self.root_loader)

        # Train loader for benign training
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=cfg["batch_size"], shuffle=True
        )
        self.train_iter = mit.seekable(self.train_loader)

    def _prepare_poisoned_data(self):
        """Create poisoned datasets for backdoor attacks."""
        cfg = self.config
        self.poisoned_iters = {}

        # Global poisoned dataset (for BFL)
        poisoned_ds = copy.deepcopy(self.train_dataset)
        base_idxs = (poisoned_ds.targets == cfg["base_class"]).nonzero().flatten().tolist()
        poison_dataset(
            poisoned_ds, cfg["dataset"], cfg["base_class"], cfg["target_class"],
            poison_frac=1.0, pattern_type=cfg["pattern_type"],
            data_idxs=base_idxs, poison_all=True
        )
        poisoned_loader = DataLoader(
            DatasetSplit(poisoned_ds, base_idxs),
            batch_size=cfg["batch_size"], shuffle=True
        )
        self.poisoned_iters["global"] = mit.seekable(poisoned_loader)

        # Sub-trigger datasets (for DBA)
        sub_iters = []
        num_sub = 4 if cfg["dataset"] == "cifar10" else 3
        for sub_idx in range(num_sub):
            sub_ds = copy.deepcopy(self.train_dataset)
            sub_base_idxs = (sub_ds.targets == cfg["base_class"]).nonzero().flatten().tolist()
            poison_dataset(
                sub_ds, cfg["dataset"], cfg["base_class"], cfg["target_class"],
                poison_frac=0.5, pattern_type=cfg["pattern_type"],
                data_idxs=sub_base_idxs, poison_all=False, agent_idx=sub_idx
            )
            sub_loader = DataLoader(
                DatasetSplit(sub_ds, sub_base_idxs),
                batch_size=cfg["batch_size"], shuffle=True
            )
            sub_iters.append(mit.seekable(sub_loader))
        self.poisoned_iters["sub_triggers"] = sub_iters

        # Poisoned eval loader (for measuring backdoor accuracy)
        poisoned_val = copy.deepcopy(self.test_dataset)
        val_base_idxs = (poisoned_val.targets == cfg["base_class"]).nonzero().flatten().tolist()
        poison_dataset(
            poisoned_val, cfg["dataset"], cfg["base_class"], cfg["target_class"],
            poison_frac=1.0, pattern_type=cfg["pattern_type"],
            data_idxs=val_base_idxs, poison_all=True
        )
        self.poisoned_val_loader = DataLoader(
            DatasetSplit(poisoned_val, val_base_idxs),
            batch_size=cfg["batch_size"], shuffle=False
        )

    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Compressed state dimension
        state, _ = get_compressed_state(self.model, num_tail_layers=2)
        self.state_dim = len(state)

        # Both agents observe the compressed global model state
        self.observation_spaces = {
            "defender": Box(-1, 1, (self.state_dim,), dtype=np.float32),
            "attacker": Box(-1, 1, (self.state_dim,), dtype=np.float32),
        }

        # Defender: (alpha, beta, epsilon) all in [-1, 1], scaled internally
        self.action_spaces = {
            "defender": Box(-1, 1, (3,), dtype=np.float32),
            "attacker": Box(-1, 1, (3,), dtype=np.float32),
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def opponent_types(self):
        """Available attack types for meta-learning."""
        return ["IPM", "LMP", "BFL", "DBA", "RL", "BRL"]

    def reset(self, seed=None, options=None):
        """
        Reset the FL environment.

        options:
            attack_type: str — which attack to use this episode
        """
        self.agents = self.possible_agents[:]

        # Restore initial model weights
        set_parameters(self.model, copy.deepcopy(self.initial_weights))
        self.current_weights = copy.deepcopy(self.initial_weights)
        self.round = 0

        # Set attack type
        if options and "attack_type" in options:
            attack_type = options["attack_type"]
        else:
            attack_type = random.choice(self.opponent_types)

        self._attack_type = attack_type
        self._attack_strategy = create_attack(attack_type)

        # Reset data iterators
        self.train_iter = mit.seekable(self.train_loader)
        self.root_iter = mit.seekable(self.root_loader)
        for key, it in self.poisoned_iters.items():
            if isinstance(it, list):
                for sub_it in it:
                    sub_it.seek(0)
            else:
                it.seek(0)

        # Get initial observation
        obs = self._get_obs()
        observations = {agent: obs for agent in self.agents}
        infos = {agent: {"attack_type": attack_type} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Execute one FL round.

        Args:
            actions: dict with "defender" and "attacker" actions (numpy arrays)
        """
        cfg = self.config
        old_weights = self.current_weights

        # 1. Parse defender action: scale from [-1, 1] to actual ranges
        def_action = np.array(actions["defender"], dtype=np.float32)
        alpha = float(def_action[0]) * 4.95 + 5.05    # [0.1, 10.0]
        beta = float(def_action[1]) * 0.225 + 0.225    # [0.0, 0.45]
        epsilon = float(def_action[2]) * 4.95 + 5.05   # [0.1, 10.0]

        # For pruning, scale differently
        if cfg["post_defense"] == "pruning":
            epsilon = float(def_action[2]) * 0.25 + 0.25  # [0.0, 0.5]

        # 2. Sample clients
        num_sampled = max(1, int(cfg["num_clients"] * cfg["subsample_rate"]))
        random.seed(self.round + cfg["seed"] * 1000)
        sampled_clients = random.sample(range(cfg["num_clients"]), num_sampled)

        selected_untargeted = [c for c in sampled_clients if c in self.untargeted_att_ids]
        selected_backdoor = [c for c in sampled_clients if c in self.backdoor_att_ids]
        selected_attackers = selected_untargeted + selected_backdoor
        benign_clients = [c for c in sampled_clients if c not in self.all_att_ids]

        # 3. Benign client training
        benign_weights = []
        for cid in benign_clients:
            set_parameters(self.model, old_weights)
            self.model.to(DEVICE)
            train(self.model, self.train_iter, epochs=1, lr=cfg["lr"])
            benign_weights.append(get_parameters(self.model))

        # 4. Execute attack
        all_weights = list(benign_weights)
        if selected_attackers:
            env_state = {
                "old_weights": old_weights,
                "benign_weights": benign_weights,
                "model": self.model,
                "device": DEVICE,
                "lr": cfg["lr"],
                "poisoned_train_iters": self.poisoned_iters,
                "attacker_train_iter": self.train_iter,
                "selected_attacker_ids": selected_attackers,
            }
            att_action = actions.get("attacker")
            malicious_weights = self._attack_strategy.execute(env_state, att_action)
            all_weights.extend(malicious_weights)

        # 5. Aggregation: clip + trimmed mean
        if len(all_weights) > 0:
            new_weights = self._aggregate(old_weights, all_weights, alpha, beta)
        else:
            new_weights = old_weights

        # 6. Update global model
        self.current_weights = new_weights
        set_parameters(self.model, new_weights)

        # 7. Evaluate (with post-training defense applied to a copy)
        defended_model = apply_post_defense(
            self.model, cfg["post_defense"], epsilon,
            eval_loader=self.root_loader, device=DEVICE
        )
        defended_model.to(DEVICE)
        defended_model.eval()

        # Main task accuracy
        _, main_acc = test(defended_model, self.test_loader)

        # Backdoor accuracy
        _, backdoor_acc = test(defended_model, self.poisoned_val_loader)

        # 8. Compute rewards
        # Defender wants high accuracy, low backdoor
        def_reward = main_acc - backdoor_acc

        # Attacker reward depends on attack type
        if self._attack_strategy and not self._attack_strategy.is_adaptive:
            # Non-adaptive: reward based on attack success
            if self._attack_type in ["BFL", "DBA", "BRL"]:
                att_reward = backdoor_acc - main_acc  # wants high backdoor, low acc
            else:
                att_reward = -main_acc  # untargeted: wants low accuracy
        else:
            # Adaptive: same structure
            if self._attack_type in ["BRL"]:
                att_reward = backdoor_acc - main_acc
            else:
                att_reward = -main_acc

        # 9. Advance round
        self.round += 1
        done = self.round >= cfg["fl_rounds"]

        # 10. Get new observation
        obs = self._get_obs()
        observations = {agent: obs for agent in self.agents}
        rewards = {"defender": def_reward, "attacker": att_reward}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {
            agent: {
                "round": self.round,
                "main_acc": main_acc,
                "backdoor_acc": backdoor_acc,
                "attack_type": self._attack_type,
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _aggregate(self, old_weights, all_weights, alpha, beta):
        """
        Aggregation pipeline: clip gradients to norm alpha,
        then coordinate-wise trimmed mean with rate beta.
        """
        old_vec = weights_to_vector(old_weights)

        # Clip gradients
        clipped_grads = []
        for w in all_weights:
            grad_vec = old_vec - weights_to_vector(w)
            norm = np.linalg.norm(grad_vec)
            if norm > alpha:
                grad_vec = grad_vec * (alpha / norm)
            clipped_grads.append(grad_vec)

        # Trimmed mean
        grads_array = np.array(clipped_grads)
        n = len(clipped_grads)
        trim_count = max(0, int(beta * n / 2))

        if trim_count > 0 and n > 2 * trim_count:
            sorted_grads = np.sort(grads_array, axis=0)
            trimmed = sorted_grads[trim_count:n - trim_count]
            mean_grad = np.mean(trimmed, axis=0)
        else:
            mean_grad = np.mean(grads_array, axis=0)

        new_weights_vec = old_vec - mean_grad
        return vector_to_weights(new_weights_vec, old_weights)

    def _get_obs(self):
        """Get compressed state observation."""
        state, _ = get_compressed_state(self.model, num_tail_layers=2)
        return state

    def render(self):
        if self.render_mode == "human":
            print(f"Round {self.round}/{self.config['fl_rounds']} "
                  f"Attack: {self._attack_type}")
