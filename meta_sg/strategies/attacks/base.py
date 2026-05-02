"""Abstract base for all attack strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from meta_sg.simulation.types import Weights
from meta_sg.strategies.types import AttackDecision, AttackType


class AttackStrategy(ABC):
    """
    All attackers implement this interface.
    execute() is called once per FL round for each malicious client's slot.
    """

    def __init__(self, attack_type: AttackType) -> None:
        self.attack_type = attack_type

    @abstractmethod
    def execute(
        self,
        old_weights: Weights,
        benign_weights: List[Weights],
        decision: AttackDecision,
        num_malicious: int = 1,
    ) -> List[Weights]:
        """
        Craft malicious weight updates for this round.

        Args:
            old_weights:    Global model W_t before this round.
            benign_weights: List of benign client updates this round.
            decision:       Decoded attacker action for this round.
            num_malicious:  How many malicious weight vectors to return.

        Returns:
            List of malicious Weights, one per malicious client slot.
        """

    def observe(
        self,
        old_weights: Weights,
        new_weights: Weights,
        decision: AttackDecision,
        summary_info: dict,
    ) -> None:
        """Optional: stateful attackers update internal state after each round."""
