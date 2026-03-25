"""
Attack implementations for adversarial Federated Learning.

Non-adaptive attacks (fixed strategy, ignore attacker action):
  - IPM: Inner Product Manipulation
  - LMP: Local Model Poisoning (against Median/Krum)
  - BFL: Backdoor FL (train on globally poisoned data)
  - DBA: Distributed Backdoor Attack (train on sub-trigger data)

Adaptive attacks (parameterized by attacker action):
  - RL: RL-based untargeted model poisoning
  - BRL: RL-based backdoor attack
"""

import numpy as np
import copy
import random
import torch
import more_itertools as mit

from .utils.fl_utils import (
    get_parameters, set_parameters, train, weights_to_vector,
    vector_to_weights, average,
)


def _craft_ipm(old_weights, benign_avg_weights, scaling=2.0):
    """
    Inner Product Manipulation attack.
    Crafted weight = old - scaling * (old - benign_avg) * (-1)
                   = old + scaling * (old - benign_avg)
    """
    weight_diff = [w1 - w2 for w1, w2 in zip(old_weights, benign_avg_weights)]
    crafted_diff = [scaling * diff * (-1) for diff in weight_diff]
    return [w1 - w2 for w1, w2 in zip(old_weights, crafted_diff)]


def _craft_lmp_median(old_weights, benign_weights_list):
    """
    Local Model Poisoning attack optimized against Median aggregation.
    Crafts weights at the extremes of benign weight distributions.
    """
    # Compute benign aggregate direction
    num_benign = len(benign_weights_list)
    if num_benign == 0:
        return old_weights

    benign_avg = [
        np.mean([w[layer] for w in benign_weights_list], axis=0)
        for layer in range(len(old_weights))
    ]
    sign = [np.sign(u - v) for u, v in zip(benign_avg, old_weights)]

    # Find min/max of benign weights per parameter
    max_weight = weights_to_vector(benign_weights_list[0]).copy()
    min_weight = weights_to_vector(benign_weights_list[0]).copy()
    for w in benign_weights_list[1:]:
        vec = weights_to_vector(w)
        max_weight = np.maximum(max_weight, vec)
        min_weight = np.minimum(min_weight, vec)

    # Craft weights at extremes opposite to gradient direction
    b = 2.0
    crafted = []
    count = 0
    for layer in sign:
        new_params = []
        for param in layer.flatten():
            if param == -1.0 and max_weight[count] > 0:
                new_params.append(random.uniform((b - 1) * max_weight[count],
                                                  b * max_weight[count]))
            elif param == -1.0:
                new_params.append(random.uniform(max_weight[count] / (b - 1),
                                                  max_weight[count]) / b)
            elif param == 1.0 and min_weight[count] > 0:
                new_params.append(random.uniform(min_weight[count] / b,
                                                  min_weight[count]) / (b - 1))
            elif param == 1.0:
                new_params.append(random.uniform(b * min_weight[count],
                                                  (b - 1) * min_weight[count]))
            elif param == 0.0:
                new_params.append(0)
            else:
                new_params.append(random.uniform(min_weight[count],
                                                  max_weight[count]))
            count += 1
        crafted.append(np.array(new_params).reshape(layer.shape))

    return crafted


class AttackStrategy:
    """Base class for attack strategies."""

    def __init__(self, attack_type, config=None):
        self.attack_type = attack_type
        self.config = config or {}
        self.is_adaptive = False

    def execute(self, env_state, attacker_action=None):
        """
        Execute the attack and return malicious weights for each attacker.

        Args:
            env_state: dict with keys:
                - old_weights: global model weights before this round
                - benign_weights: list of benign client weights this round
                - model: the global model (nn.Module)
                - poisoned_train_iters: dict of poisoned data iterators
                - selected_attacker_ids: list of selected attacker client ids
                - device: torch device
                - lr: learning rate
            attacker_action: numpy array of attacker actions (for adaptive attacks)

        Returns:
            list of malicious weight arrays, one per selected attacker
        """
        raise NotImplementedError


class IPMAttack(AttackStrategy):
    """Inner Product Manipulation — non-adaptive untargeted."""

    def __init__(self, scaling=2.0):
        super().__init__("IPM", {"scaling": scaling})

    def execute(self, env_state, attacker_action=None):
        old_w = env_state["old_weights"]
        benign_w = env_state["benign_weights"]
        num_attackers = len(env_state["selected_attacker_ids"])

        if len(benign_w) == 0:
            return [old_w] * num_attackers

        # Average benign weights
        benign_avg = [
            np.mean([w[l] for w in benign_w], axis=0)
            for l in range(len(old_w))
        ]

        crafted = _craft_ipm(old_w, benign_avg, self.config["scaling"])
        return [crafted] * num_attackers


class LMPAttack(AttackStrategy):
    """Local Model Poisoning — non-adaptive untargeted (against Median)."""

    def __init__(self):
        super().__init__("LMP")

    def execute(self, env_state, attacker_action=None):
        old_w = env_state["old_weights"]
        benign_w = env_state["benign_weights"]
        num_attackers = len(env_state["selected_attacker_ids"])

        if len(benign_w) == 0:
            return [old_w] * num_attackers

        crafted = _craft_lmp_median(old_w, benign_w)
        return [crafted] * num_attackers


class BFLAttack(AttackStrategy):
    """Backdoor FL — non-adaptive targeted. Trains on globally poisoned data."""

    def __init__(self, poison_frac=1.0):
        super().__init__("BFL", {"poison_frac": poison_frac})

    def execute(self, env_state, attacker_action=None):
        old_w = env_state["old_weights"]
        model = env_state["model"]
        device = env_state["device"]
        lr = env_state["lr"]
        poisoned_iter = env_state["poisoned_train_iters"].get("global")
        num_attackers = len(env_state["selected_attacker_ids"])

        if poisoned_iter is None:
            return [old_w] * num_attackers

        malicious_weights = []
        for _ in range(num_attackers):
            set_parameters(model, old_w)
            model.to(device)
            train(model, poisoned_iter, epochs=1, lr=lr)
            malicious_weights.append(get_parameters(model))

        return malicious_weights


class DBAAttack(AttackStrategy):
    """Distributed Backdoor Attack — non-adaptive targeted. Sub-triggers."""

    def __init__(self, num_sub_triggers=4, poison_frac=0.5):
        super().__init__("DBA", {
            "num_sub_triggers": num_sub_triggers,
            "poison_frac": poison_frac,
        })

    def execute(self, env_state, attacker_action=None):
        old_w = env_state["old_weights"]
        model = env_state["model"]
        device = env_state["device"]
        lr = env_state["lr"]
        sub_iters = env_state["poisoned_train_iters"].get("sub_triggers", [])
        num_attackers = len(env_state["selected_attacker_ids"])
        num_subs = self.config["num_sub_triggers"]

        if len(sub_iters) == 0:
            return [old_w] * num_attackers

        malicious_weights = []
        for i in range(num_attackers):
            set_parameters(model, old_w)
            model.to(device)
            sub_idx = random.randint(0, min(num_subs, len(sub_iters)) - 1)
            train(model, sub_iters[sub_idx], epochs=1, lr=lr)
            malicious_weights.append(get_parameters(model))

        return malicious_weights


class RLAttack(AttackStrategy):
    """RL-based untargeted model poisoning — adaptive."""

    def __init__(self):
        super().__init__("RL")
        self.is_adaptive = True

    def execute(self, env_state, attacker_action=None):
        """
        attacker_action: [poison_frac_scaled, lr_scaled, epochs_scaled]
        All in [-1, 1], scaled to actual ranges.
        """
        old_w = env_state["old_weights"]
        benign_w = env_state["benign_weights"]
        model = env_state["model"]
        device = env_state["device"]
        num_attackers = len(env_state["selected_attacker_ids"])
        train_iter = env_state.get("attacker_train_iter")

        if attacker_action is None or len(benign_w) == 0:
            return [old_w] * num_attackers

        # Scale actions from [-1, 1] to actual ranges
        scaling = float(attacker_action[0]) * 5.0 + 5.0    # [0, 10]
        local_lr = float(attacker_action[1]) * 0.05 + 0.05  # [0, 0.1]
        local_epochs = int(float(attacker_action[2]) * 5 + 6)  # [1, 11]
        local_epochs = max(1, min(11, local_epochs))

        # Train on clean/attacker data, then craft
        set_parameters(model, old_w)
        model.to(device)
        if train_iter is not None:
            train(model, train_iter, epochs=local_epochs, lr=local_lr)

        att_weights = get_parameters(model)

        # Craft: amplify the difference
        benign_avg = [
            np.mean([w[l] for w in benign_w], axis=0)
            for l in range(len(old_w))
        ]
        crafted = _craft_ipm(old_w, att_weights, scaling)
        return [crafted] * num_attackers


class BRLAttack(AttackStrategy):
    """RL-based backdoor attack — adaptive."""

    def __init__(self):
        super().__init__("BRL")
        self.is_adaptive = True

    def execute(self, env_state, attacker_action=None):
        """
        attacker_action: [poison_frac_scaled, lr_scaled, epochs_scaled]
        """
        old_w = env_state["old_weights"]
        model = env_state["model"]
        device = env_state["device"]
        num_attackers = len(env_state["selected_attacker_ids"])
        poisoned_iter = env_state["poisoned_train_iters"].get("global")

        if attacker_action is None or poisoned_iter is None:
            return [old_w] * num_attackers

        # Scale actions
        local_lr = float(attacker_action[1]) * 0.05 + 0.05
        local_epochs = int(float(attacker_action[2]) * 5 + 6)
        local_epochs = max(1, min(11, local_epochs))

        malicious_weights = []
        for _ in range(num_attackers):
            set_parameters(model, old_w)
            model.to(device)
            train(model, poisoned_iter, epochs=local_epochs, lr=local_lr)
            malicious_weights.append(get_parameters(model))

        return malicious_weights


# Registry for easy lookup
ATTACK_REGISTRY = {
    "IPM": IPMAttack,
    "LMP": LMPAttack,
    "BFL": BFLAttack,
    "DBA": DBAAttack,
    "RL": RLAttack,
    "BRL": BRLAttack,
}


def create_attack(attack_type: str, **kwargs) -> AttackStrategy:
    """Factory function to create attack strategies."""
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack type: {attack_type}. "
                         f"Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_type](**kwargs)
