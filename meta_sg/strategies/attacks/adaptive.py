"""
Adaptive (RL-based) attack strategy.
Wraps a TD3Agent that outputs attack decisions from the current state observation.
Used in meta-SG inner loop (attacker best-response).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from meta_sg.simulation.types import Weights
from meta_sg.strategies.attacks.base import AttackStrategy
from meta_sg.strategies.types import AttackDecision, AttackType, ATTACK_DOMAIN

if TYPE_CHECKING:
    from meta_sg.learning.td3 import TD3Agent
    from meta_sg.games.observations import compress_weights


def _weights_to_vec(weights: Weights) -> np.ndarray:
    return np.concatenate([w.ravel() for w in weights])


def _vec_to_weights(vec: np.ndarray, template: Weights) -> Weights:
    result, idx = [], 0
    for w in template:
        n = w.size
        result.append(vec[idx: idx + n].reshape(w.shape))
        idx += n
    return result


class AdaptiveAttackStrategy(AttackStrategy):
    """
    RL-based attack — paper §III-A.
    Policy π_ξ(a_A | s; φ) is a TD3 actor.
    The attacker observes the same compressed state as the defender.
    """

    def __init__(
        self,
        attack_type: AttackType,
        agent: "TD3Agent",
        noise: float = 0.1,
    ) -> None:
        super().__init__(attack_type)
        self.agent = agent
        self.noise = noise
        self._last_obs: Optional[np.ndarray] = None

    def set_obs(self, obs: np.ndarray) -> None:
        """Called by BSMGEnv before execute() with the current state."""
        self._last_obs = obs

    def get_raw_action(self, obs: Optional[np.ndarray] = None) -> np.ndarray:
        state = obs if obs is not None else self._last_obs
        if state is None:
            return np.zeros(3, dtype=np.float32)
        return self.agent.get_action(state, noise=self.noise)

    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        """
        Craft malicious updates using the decoded AttackDecision.
        For the stub: IPM-style with RL-tuned gamma_scale.
        In full implementation: local_search_update with stealth regularisation.
        """
        if not benign_weights:
            return [old_weights] * num_malicious

        old_vec = _weights_to_vec(old_weights)
        benign_vecs = np.stack([_weights_to_vec(w) for w in benign_weights])
        mean_update = np.mean(benign_vecs - old_vec, axis=0)

        # Stealth regularisation: blend between IPM and benign mean
        gamma = decision.gamma_scale
        stealth = decision.lambda_stealth
        malicious_direction = -gamma * mean_update
        # Stealth blend: pull closer to benign mean to evade detection
        blend = (1.0 - stealth) * malicious_direction + stealth * mean_update
        malicious_vec = old_vec + blend

        malicious = _vec_to_weights(malicious_vec, old_weights)
        return [malicious] * num_malicious
