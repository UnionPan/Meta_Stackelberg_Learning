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
