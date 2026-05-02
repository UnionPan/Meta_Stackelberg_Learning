"""Abstract base for all defense strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from meta_sg.simulation.types import Weights
from meta_sg.strategies.types import DefenseDecision


class DefenseStrategy(ABC):
    """
    Defense interface — two distinct stages as per the paper:

    Stage 1 (in coordinator):  aggregate() produces W_{t+1} for next round.
    Stage 2 (in game layer):   apply_post_training() modifies a COPY of W_{t+1}
                               solely for reward evaluation; never touches W_{t+1}.
    """

    @abstractmethod
    def aggregate(
        self,
        old_weights: Weights,
        all_weights: List[Weights],
        decision: DefenseDecision,
        trusted_weights: Optional[Weights] = None,
    ) -> Weights:
        """
        Aggregate client updates -> new global model W_{t+1}.
        This is the ONLY place that determines the next FL state.
        """

    def apply_post_training(
        self,
        weights: Weights,
        decision: DefenseDecision,
    ) -> Weights:
        """
        Apply post-training defense (NeuroClip / Prun) to a weight COPY.
        Result is used only for reward computation (Remark 1 in paper).
        Default: identity (no post-training defense).
        """
        return [w.copy() for w in weights]
