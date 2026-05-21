"""Observation construction for TD3 backdoor policy learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_attacker.observation import FixedRandomProjector


def weights_to_vector(weights) -> np.ndarray:
    if weights is None:
        return np.zeros(1, dtype=np.float32)
    arrays = [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in weights]
    if not arrays:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32)


def tail_weights_to_vector(weights, *, tail_layers: int = 2) -> np.ndarray:
    if weights is None:
        return np.zeros(1, dtype=np.float32)
    layers = list(weights)[-max(1, int(tail_layers)) :]
    arrays = [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in layers]
    if not arrays:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32)


def normalize_and_clip(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(1, dtype=np.float32)
    values = (values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)
    return np.clip(values, -5.0, 5.0).astype(np.float32)


@dataclass
class BackdoorObservationBuilder:
    """Builds fixed-size state for a TD3 backdoor policy.

    Each step stores projected tail-layer parameters, projected tail-layer
    delta, the last raw TD3 action, attacker sampling information, and
    paper-level feedback: progress, clean accuracy, and attack success rate.
    """

    config: BackdoorRLConfig
    history: list[np.ndarray] = field(default_factory=list)
    projector: FixedRandomProjector | None = None

    def __post_init__(self) -> None:
        if self.projector is None:
            self.projector = FixedRandomProjector(self.config.projection_dim, self.config.seed)

    def reset(self) -> None:
        self.history.clear()

    def build(
        self,
        *,
        weights,
        previous_weights,
        last_action,
        round_idx: int,
        total_rounds: int,
        sampled_attacker_count: int | float,
        num_attackers: int | float,
        sampled_client_count: int | float,
        clean_acc: float,
        asr: float,
    ) -> np.ndarray:
        current = tail_weights_to_vector(weights)
        previous = tail_weights_to_vector(previous_weights)
        if previous.shape != current.shape:
            previous = np.zeros_like(current)
        action = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if action.size < self.config.action_dim:
            padded = np.zeros(self.config.action_dim, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        action = np.clip(action[: self.config.action_dim], -1.0, 1.0).astype(np.float32)
        attacker_over_attackers = float(sampled_attacker_count) / max(1.0, float(num_attackers))
        attacker_over_sampled = float(sampled_attacker_count) / max(1.0, float(sampled_client_count))
        feedback = np.asarray(
            [
                float(np.clip(attacker_over_attackers, 0.0, 1.0)),
                float(np.clip(attacker_over_sampled, 0.0, 1.0)),
                float(round_idx) / max(1.0, float(total_rounds)),
                float(np.nan_to_num(clean_acc, nan=0.0)),
                float(np.nan_to_num(asr, nan=0.0)),
            ],
            dtype=np.float32,
        )
        step_obs = np.concatenate(
            [
                self.projector.project(normalize_and_clip(current)),
                self.projector.project(normalize_and_clip(current - previous)),
                action,
                feedback,
            ],
            axis=0,
        ).astype(np.float32)
        self.history.append(step_obs)
        self.history = self.history[-max(1, int(self.config.history_window)) :]
        padded = list(self.history)
        while len(padded) < max(1, int(self.config.history_window)):
            padded.insert(0, np.zeros(self.config.per_step_observation_dim, dtype=np.float32))
        return np.concatenate(padded, axis=0).astype(np.float32)
