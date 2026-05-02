"""
Non-adaptive attack strategies: IPM, LMP, BFL, DBA.
Paper Table 3: all non-adaptive.
"""
from __future__ import annotations

from typing import List

import numpy as np

from meta_sg.simulation.types import Weights
from meta_sg.strategies.attacks.base import AttackStrategy
from meta_sg.strategies.types import AttackDecision, AttackType, ATTACK_DOMAIN


def _weights_to_vec(weights: Weights) -> np.ndarray:
    return np.concatenate([w.ravel() for w in weights])


def _vec_to_weights(vec: np.ndarray, template: Weights) -> Weights:
    result, idx = [], 0
    for w in template:
        n = w.size
        result.append(vec[idx: idx + n].reshape(w.shape))
        idx += n
    return result


class IPMAttack(AttackStrategy):
    """
    Inner Product Manipulation (IPM) [Xie et al. 2020].
    Malicious update = -γ * mean_benign_update (negates benign gradient direction).
    Paper: scaling factor γ=2 by default.
    """

    def __init__(self, scaling: float = 2.0) -> None:
        super().__init__(ATTACK_DOMAIN["ipm"])
        self.scaling = scaling

    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        if not benign_weights:
            return [old_weights] * num_malicious

        old_vec = _weights_to_vec(old_weights)
        benign_vecs = np.stack([_weights_to_vec(w) for w in benign_weights])
        mean_update = np.mean(benign_vecs - old_vec, axis=0)

        gamma = self.scaling * decision.gamma_scale
        malicious_vec = old_vec - gamma * mean_update
        malicious = _vec_to_weights(malicious_vec, old_weights)
        return [malicious] * num_malicious


class LMPAttack(AttackStrategy):
    """
    Local Model Poisoning (LMP) [Fang et al. 2020].
    Amplifies the deviation from the current global model.
    """

    def __init__(self, scale: float = 5.0) -> None:
        super().__init__(ATTACK_DOMAIN["lmp"])
        self.scale = scale

    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        if not benign_weights:
            return [old_weights] * num_malicious

        old_vec = _weights_to_vec(old_weights)
        benign_vecs = np.stack([_weights_to_vec(w) for w in benign_weights])
        mean_benign = np.mean(benign_vecs, axis=0)

        scale = self.scale * decision.gamma_scale
        # Push model in opposite direction of benign mean
        malicious_vec = old_vec + scale * (old_vec - mean_benign)
        malicious = _vec_to_weights(malicious_vec, old_weights)
        return [malicious] * num_malicious


class BFLAttack(AttackStrategy):
    """
    Backdoor Federated Learning (BFL) [Bagdasaryan et al. 2020].
    Scales up poisoned model to survive aggregation.
    In stub: shifts weights toward a fixed direction to simulate backdoor injection.
    """

    def __init__(self, poison_frac: float = 1.0, boost_factor: float = 3.0) -> None:
        super().__init__(ATTACK_DOMAIN["bfl"])
        self.poison_frac = poison_frac
        self.boost_factor = boost_factor

    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        old_vec = _weights_to_vec(old_weights)
        rng = np.random.default_rng(seed=42)
        backdoor_direction = rng.standard_normal(old_vec.shape).astype(np.float32)
        backdoor_direction /= np.linalg.norm(backdoor_direction) + 1e-8

        boost = self.boost_factor * decision.gamma_scale
        malicious_vec = old_vec + boost * backdoor_direction
        malicious = _vec_to_weights(malicious_vec, old_weights)
        return [malicious] * num_malicious


class DBAAttack(AttackStrategy):
    """
    Distributed Backdoor Attack (DBA) [Xie et al. 2019].
    Each malicious client injects a sub-trigger; combined triggers form the full backdoor.
    In stub: each malicious client injects a different directional perturbation.
    """

    def __init__(self, poison_frac: float = 0.5, num_sub_triggers: int = 4) -> None:
        super().__init__(ATTACK_DOMAIN["dba"])
        self.poison_frac = poison_frac
        self.num_sub_triggers = num_sub_triggers

    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        old_vec = _weights_to_vec(old_weights)
        result = []
        for i in range(num_malicious):
            rng = np.random.default_rng(seed=i % self.num_sub_triggers)
            direction = rng.standard_normal(old_vec.shape).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            malicious_vec = old_vec + decision.gamma_scale * direction
            result.append(_vec_to_weights(malicious_vec, old_weights))
        return result


def build_fixed_attack(attack_type: "AttackType") -> AttackStrategy:
    name = attack_type.name.lower()
    cfg = attack_type.config
    if name == "ipm":
        return IPMAttack(scaling=cfg.get("scaling", 2.0))
    if name == "lmp":
        return LMPAttack(scale=cfg.get("scale", 5.0))
    if name == "bfl":
        return BFLAttack(poison_frac=cfg.get("poison_frac", 1.0))
    if name == "dba":
        return DBAAttack(
            poison_frac=cfg.get("poison_frac", 0.5),
            num_sub_triggers=cfg.get("num_sub_triggers", 4),
        )
    raise ValueError(f"Unknown fixed attack type: {name}")
