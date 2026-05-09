"""Observation construction for the adaptive RL attacker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fl_sandbox.attacks.base import set_model_weights
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.models import get_compressed_state

DEFENSES = (
    "fedavg",
    "krum",
    "multi_krum",
    "median",
    "clipped_median",
    "geometric_median",
    "trimmed_mean",
    "fltrust",
    "paper_norm_trimmed_mean",
)


def _weights_to_vector(weights) -> np.ndarray:
    if weights is None:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate([np.asarray(layer, dtype=np.float32).reshape(-1) for layer in weights]).astype(np.float32)


@dataclass
class FixedRandomProjector:
    """Seeded random projection that is fixed after first input dimension."""

    output_dim: int
    seed: int
    _matrix: np.ndarray | None = None

    def project(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._matrix is None or self._matrix.shape[1] != vector.shape[0]:
            rng = np.random.default_rng(self.seed)
            scale = 1.0 / np.sqrt(max(1, vector.shape[0]))
            self._matrix = rng.normal(0.0, scale, size=(self.output_dim, vector.shape[0])).astype(np.float32)
        return (self._matrix @ vector).astype(np.float32)


@dataclass
class ProjectedObservationBuilder:
    """Canonical MDP observation builder with H-step history."""

    config: RLAttackerConfig
    action_dim: int
    defenses: tuple[str, ...] = DEFENSES
    history: list[np.ndarray] = field(default_factory=list)
    projector: FixedRandomProjector | None = None

    def __post_init__(self) -> None:
        if self.projector is None:
            self.projector = FixedRandomProjector(self.config.projection_dim, self.config.seed)

    @property
    def per_step_dim(self) -> int:
        return 3 * int(self.config.projection_dim) + int(self.action_dim) + 2 + len(self.defenses)

    def build(
        self,
        *,
        weights,
        previous_weights,
        last_aggregate_update,
        last_action: np.ndarray,
        last_bypass_score: float,
        round_idx: int,
        total_rounds: int,
        defense_type: str,
    ) -> np.ndarray:
        current_vec = _weights_to_vector(weights)
        previous_vec = _weights_to_vector(previous_weights)
        if previous_vec.shape != current_vec.shape:
            previous_vec = np.zeros_like(current_vec)
        update_vec = _weights_to_vector(last_aggregate_update)
        if update_vec.shape != current_vec.shape:
            update_vec = current_vec - previous_vec
        action = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded
        action = action[: self.action_dim]
        defense_one_hot = np.zeros(len(self.defenses), dtype=np.float32)
        defense = defense_type.lower()
        if defense in self.defenses:
            defense_one_hot[self.defenses.index(defense)] = 1.0
        step_obs = np.concatenate(
            [
                self.projector.project(current_vec),
                self.projector.project(current_vec - previous_vec),
                self.projector.project(update_vec),
                action,
                np.asarray([float(last_bypass_score), float(round_idx) / max(1.0, float(total_rounds))], dtype=np.float32),
                defense_one_hot,
            ],
            axis=0,
        ).astype(np.float32)
        self.history.append(step_obs)
        self.history = self.history[-max(1, int(self.config.history_window)) :]
        padded_history = list(self.history)
        while len(padded_history) < max(1, int(self.config.history_window)):
            padded_history.insert(0, np.zeros(self.per_step_dim, dtype=np.float32))
        return np.concatenate(padded_history, axis=0).astype(np.float32)


@dataclass
class ObservationBuilder:
    config: RLAttackerConfig
    history: list[np.ndarray] = field(default_factory=list)
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    previous_bypass_score: float = 0.0

    def build(
        self,
        *,
        model_template,
        weights,
        device,
        defense_type: str,
        round_idx: int,
        total_rounds: int,
        num_attackers: int,
        max_attackers: int,
    ) -> np.ndarray:
        model = model_template
        set_model_weights(model, weights, device)
        compressed, _ = get_compressed_state(model, num_tail_layers=self.config.state_tail_layers)
        compressed = np.asarray(compressed, dtype=np.float32).reshape(-1)
        self.history.append(compressed)
        self.history = self.history[-max(1, self.config.history_window):]
        history = list(self.history)
        while len(history) < max(1, self.config.history_window):
            history.insert(0, np.zeros_like(compressed))
        defense_one_hot = np.zeros(len(DEFENSES), dtype=np.float32)
        if defense_type.lower() in DEFENSES:
            defense_one_hot[DEFENSES.index(defense_type.lower())] = 1.0
        round_norm = np.asarray([float(round_idx) / max(1.0, float(total_rounds))], dtype=np.float32)
        attacker_norm = np.asarray([0.0], dtype=np.float32)
        if max_attackers > 0:
            attacker_norm[0] = 2.0 * (float(num_attackers) / float(max_attackers)) - 1.0
        return np.concatenate(
            [
                *history,
                self.previous_action.astype(np.float32).reshape(-1),
                np.asarray([self.previous_bypass_score], dtype=np.float32),
                round_norm,
                attacker_norm,
                defense_one_hot,
            ],
            axis=0,
        ).astype(np.float32)


def build_observation_from_state(state: dict[str, Any], max_attackers: int) -> np.ndarray:
    pram = np.asarray(state["pram"], dtype=np.float32).reshape(-1)
    attacker_norm = 0.0
    if max_attackers > 0:
        attacker_norm = 2.0 * (int(state.get("num_attacker", 0)) / max_attackers) - 1.0
    return np.concatenate([pram, np.asarray([attacker_norm], dtype=np.float32)], axis=0)
