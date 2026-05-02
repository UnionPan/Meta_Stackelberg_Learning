"""Abstract interface for FL simulation coordinators (Layer 1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .types import InitialState, RoundSummary, SimulationSnapshot, SimulationSpec, Weights

if TYPE_CHECKING:
    from meta_sg.strategies.types import AttackDecision, DefenseDecision
    from meta_sg.strategies.attacks.base import AttackStrategy
    from meta_sg.strategies.defenses.base import DefenseStrategy


class FLCoordinator(ABC):
    """
    Pure FL transition executor.
    Manages W_t -> W_{t+1} via aggregate(α, β).
    Post-training defense is NOT applied here — only in the game layer for reward.
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> InitialState:
        """Initialise model weights, data partitions, attacker IDs."""

    @abstractmethod
    def run_round(
        self,
        attack: "AttackStrategy",
        defense: "DefenseStrategy",
        attack_decision: "AttackDecision",
        defense_decision: "DefenseDecision",
        evaluate: bool = True,
    ) -> RoundSummary:
        """
        Execute one FL round:
          1. Sample clients
          2. Benign clients train locally
          3. attack.execute() -> malicious weights
          4. defense.aggregate(alpha, beta) -> W_{t+1}
          5. If evaluate: compute clean_acc, backdoor_acc on W_{t+1}

        Post-training defense (NeuroClip/Prun) is NOT called here.
        """

    @property
    @abstractmethod
    def current_weights(self) -> Weights:
        """Current global model W_t (before next round)."""

    @property
    @abstractmethod
    def spec(self) -> SimulationSpec:
        """Static model/data shape metadata. Must not mutate coordinator state."""

    @abstractmethod
    def snapshot(self) -> SimulationSnapshot:
        """Serialise current state for rollback / parallel rollouts."""

    @abstractmethod
    def restore(self, snapshot: SimulationSnapshot) -> None:
        """Restore coordinator to a previously snapshotted state."""
