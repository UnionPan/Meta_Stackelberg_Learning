"""Trajectory types for BSMG rollouts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from meta_sg.strategies.types import AttackType


@dataclass
class Transition:
    """One step τ_ξ^t = (s_t, a_D^t, a_A^t, r_D^t, r_A^t, s_{t+1})."""
    state: np.ndarray
    defender_action: np.ndarray
    attacker_action: np.ndarray
    defender_reward: float
    attacker_reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    H-step trajectory τ_ξ for one attack type ξ.
    Used by MetaSGTrainer to compute policy gradients.
    """
    attack_type: AttackType
    transitions: List[Transition] = field(default_factory=list)

    @property
    def horizon(self) -> int:
        return len(self.transitions)

    def defender_returns(self, gamma: float = 0.99) -> List[float]:
        """Discounted cumulative defender rewards (Monte Carlo)."""
        returns = []
        g = 0.0
        for t in reversed(self.transitions):
            g = t.defender_reward + gamma * g
            returns.insert(0, g)
        return returns

    def attacker_returns(self, gamma: float = 0.99) -> List[float]:
        returns = []
        g = 0.0
        for t in reversed(self.transitions):
            g = t.attacker_reward + gamma * g
            returns.insert(0, g)
        return returns

    def to_defender_buffer_tuples(self):
        """Yield (s, a_D, r_D, s', done) for defender replay buffer."""
        for t in self.transitions:
            yield t.state, t.defender_action, t.defender_reward, t.next_state, t.done

    def to_attacker_buffer_tuples(self):
        """Yield (s, a_A, r_A, s', done) for attacker replay buffer."""
        for t in self.transitions:
            yield t.state, t.attacker_action, t.attacker_reward, t.next_state, t.done
