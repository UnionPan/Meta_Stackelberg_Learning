"""
Stub FL coordinator — no real PyTorch training, no datasets.
Simulates benign updates as Gaussian perturbations around current weights.
Allows testing Layers 2-4 independently of real FL (fl_sandbox).

When fl_sandbox is connected, replace StubCoordinator with FLSandboxCoordinator.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .interface import FLCoordinator
from .types import InitialState, RoundSummary, SimulationSnapshot, SimulationSpec, Weights

if TYPE_CHECKING:
    from meta_sg.strategies.types import AttackDecision, DefenseDecision
    from meta_sg.strategies.attacks.base import AttackStrategy
    from meta_sg.strategies.defenses.base import DefenseStrategy


# Default stub model architecture (approximates MNIST CNN last layers).
# Last two layer shapes: (128, 10) and (10,)  →  obs_dim = 1290
DEFAULT_LAYER_SHAPES = [
    (32, 1, 8, 8),   # conv1 filters: (out_ch, in_ch, kH, kW)
    (32,),           # conv1 bias
    (64, 32, 6, 6),  # conv2 filters
    (64,),           # conv2 bias
    (64, 64, 5, 5),  # conv3 filters
    (64,),           # conv3 bias
    (64 * 4 * 4, 128),  # FC1 weights  ← "last-2" starts here
    (128,),             # FC1 bias
    (128, 10),          # FC2 weights
    (10,),              # FC2 bias
]


class StubCoordinator(FLCoordinator):
    """
    Lightweight stub for testing Layers 2-4 without a real FL environment.

    Benign updates: Gaussian noise around current weights.
    Metrics:        Dummy scalar values that evolve plausibly over rounds.
    """

    def __init__(
        self,
        num_clients: int = 10,
        num_attackers: int = 2,
        subsample_rate: float = 1.0,
        layer_shapes: Optional[List[tuple]] = None,
        update_noise_std: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.subsample_rate = subsample_rate
        self.layer_shapes = layer_shapes or DEFAULT_LAYER_SHAPES
        self.update_noise_std = update_noise_std
        self._seed = seed

        self._rng = np.random.default_rng(seed)
        self._weights: Weights = []
        self._round_idx: int = 0
        self._base_clean_acc: float = 0.5
        self._last_defense_alpha: float = 0.0
        self._last_defense_beta: float = 0.0

    # ------------------------------------------------------------------
    # FLCoordinator interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> InitialState:
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        self._rng = rng
        self._weights = [
            rng.standard_normal(shape).astype(np.float32) * 0.1
            for shape in self.layer_shapes
        ]
        self._round_idx = 0
        self._base_clean_acc = 0.5
        return InitialState(weights=[w.copy() for w in self._weights], round_idx=0)

    def run_round(
        self,
        attack: "AttackStrategy",
        defense: "DefenseStrategy",
        attack_decision: "AttackDecision",
        defense_decision: "DefenseDecision",
        evaluate: bool = True,
    ) -> RoundSummary:
        old_weights = [w.copy() for w in self._weights]

        # Simulate benign client local training
        num_sampled = max(1, int(self.num_clients * self.subsample_rate))
        attacker_ids = set(range(self.num_attackers))
        benign_ids = [i for i in range(num_sampled) if i not in attacker_ids]
        selected_attackers = [i for i in range(num_sampled) if i in attacker_ids]

        benign_weights = [
            self._simulate_benign_update(old_weights) for _ in benign_ids
        ]

        # Execute attack strategy
        if selected_attackers and attack is not None:
            malicious_weights = attack.execute(
                old_weights, benign_weights, attack_decision,
                num_malicious=len(selected_attackers),
            )
        else:
            malicious_weights = []

        # Aggregate with defense strategy
        all_weights = benign_weights + malicious_weights
        if all_weights and defense is not None:
            self._weights = defense.aggregate(
                old_weights, all_weights, defense_decision
            )
            self._last_defense_alpha = float(defense_decision.norm_bound_alpha)
            self._last_defense_beta = float(defense_decision.trimmed_mean_beta)
        elif benign_weights:
            self._weights = _fedavg(benign_weights)
            self._last_defense_alpha = 0.0
            self._last_defense_beta = 0.0

        self._round_idx += 1

        # Compute dummy metrics
        clean_acc, backdoor_acc = self._compute_dummy_metrics(
            old_weights, self._weights, attack
        ) if evaluate else (self._base_clean_acc, 0.0)

        return RoundSummary(
            round_idx=self._round_idx,
            clean_acc=clean_acc,
            backdoor_acc=backdoor_acc,
            clean_loss=1.0 - clean_acc,
            attack_name=attack.attack_type.name if attack else "clean",
            defense_name=defense.__class__.__name__ if defense else "none",
        )

    @property
    def current_weights(self) -> Weights:
        return [w.copy() for w in self._weights]

    @property
    def spec(self) -> SimulationSpec:
        return SimulationSpec(layer_shapes=tuple(tuple(shape) for shape in self.layer_shapes))

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            round_idx=self._round_idx,
            weights=[w.copy() for w in self._weights],
            rng_state=copy.deepcopy(self._rng),
        )

    def restore(self, snapshot: SimulationSnapshot) -> None:
        self._round_idx = snapshot.round_idx
        self._weights = [w.copy() for w in snapshot.weights]
        if snapshot.rng_state is not None:
            self._rng = copy.deepcopy(snapshot.rng_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_benign_update(self, old_weights: Weights) -> Weights:
        """Benign client update = small gradient step (simulated as noise)."""
        return [
            w + self._rng.standard_normal(w.shape).astype(np.float32) * self.update_noise_std
            for w in old_weights
        ]

    def _compute_dummy_metrics(
        self, old_weights: Weights, new_weights: Weights, attack
    ):
        """
        Simulate plausible clean_acc / backdoor_acc.
        clean_acc: increases slowly toward 0.95 (FL convergence).
        backdoor_acc: increases if attack is targeted, depends on aggregation noise.
        """
        # Slow convergence signal with action-dependent proxy effects.
        self._base_clean_acc = min(0.95, self._base_clean_acc + 0.001)
        noise = float(self._rng.standard_normal() * 0.02)
        defense_strength = np.clip(
            (5.0 - self._last_defense_alpha) / 5.0 + self._last_defense_beta,
            0.0,
            1.5,
        )
        utility_cost = 0.035 * defense_strength
        clean_acc = float(np.clip(self._base_clean_acc - utility_cost + noise, 0.0, 1.0))

        if attack is not None and attack.attack_type.objective == "targeted":
            # Backdoor acc: grows slowly if not well-defended
            backdoor_pressure = 0.1 + self._round_idx * 0.003 + noise
            backdoor_acc = float(np.clip(backdoor_pressure - 0.06 * defense_strength, 0.0, 1.0))
        else:
            base_bd = self._rng.uniform(0.0, 0.15)
            backdoor_acc = float(np.clip(base_bd - 0.04 * defense_strength, 0.0, 1.0))

        return clean_acc, backdoor_acc


def _fedavg(weights_list: List[Weights]) -> Weights:
    return [
        np.mean([ws[i] for ws in weights_list], axis=0)
        for i in range(len(weights_list[0]))
    ]
