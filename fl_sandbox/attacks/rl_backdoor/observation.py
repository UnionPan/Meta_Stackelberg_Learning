"""Observable-only state for the RL backdoor attacker.

The same builder is used inside the grey-box simulator and on the live FL
round — the sim→real transfer precondition is that every component is
computed identically in both worlds. Privileged global metrics (clean
accuracy, true global ASR) are deliberately excluded; the attacker only sees
what it can legitimately observe in a real deployment: the global model, the
sampling counts, and its own trigger-set evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fl_sandbox.attacks.rl_attacker.observation import FixedRandomProjector
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig


def _tail_vector(weights, *, tail_layers: int) -> np.ndarray:
    if weights is None:
        return np.zeros(1, dtype=np.float32)
    tail = list(weights)[-max(1, int(tail_layers)) :]
    arrays = [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in tail]
    if not arrays:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32)


def _unit_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _full_delta(weights, previous_weights) -> np.ndarray:
    if weights is None or previous_weights is None:
        return np.zeros(1, dtype=np.float32)
    deltas = []
    for new, old in zip(weights, previous_weights):
        new_arr = np.asarray(new, dtype=np.float32).reshape(-1)
        old_arr = np.asarray(old, dtype=np.float32).reshape(-1)
        if new_arr.shape != old_arr.shape:
            continue
        deltas.append(new_arr - old_arr)
    if not deltas:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(deltas)


@dataclass
class BackdoorObservationBuilder:
    """Composes the observable-only backdoor state.

    Layout (per step, ``H=1`` by default)::

        [ round_phase = tanh(round / round_phase_tau),     # 1
          att_frac_of_attackers,                            # 1
          att_frac_of_clients,                              # 1
          local_bd_success,                                 # 1  (attacker eval
                                                            #     on own trigger
                                                            #     set, NOT global ASR)
          global_delta_lognorm = log1p(||w_t - w_{t-1}||),  # 1
          proj(unit(tail(w_t))),                            # projection_dim
          proj(unit(tail(Δw_t))),                           # projection_dim
          last_action ]                                     # action_dim
    """

    config: BackdoorRLConfig
    history: list[np.ndarray] = field(default_factory=list)
    projector: FixedRandomProjector | None = None

    def __post_init__(self) -> None:
        if self.projector is None:
            self.projector = FixedRandomProjector(self.config.projection_dim, self.config.seed)

    def reset(self) -> None:
        self.history.clear()

    @property
    def per_step_dim(self) -> int:
        return self.config.per_step_observation_dim

    def build(
        self,
        *,
        weights,
        previous_weights,
        last_action,
        round_idx: int,
        sampled_attacker_count: int | float,
        num_attackers: int | float,
        sampled_client_count: int | float,
        local_bd_success: float,
    ) -> np.ndarray:
        current_tail = _tail_vector(weights, tail_layers=self.config.state_tail_layers)
        prev_tail = _tail_vector(previous_weights, tail_layers=self.config.state_tail_layers)
        if prev_tail.shape != current_tail.shape:
            prev_tail = np.zeros_like(current_tail)
        delta_tail = current_tail - prev_tail
        delta_full = _full_delta(weights, previous_weights)

        round_phase = float(
            np.tanh(float(round_idx) / max(1.0, float(self.config.round_phase_tau)))
        )
        atk_frac_atk = float(
            np.clip(
                float(sampled_attacker_count) / max(1.0, float(num_attackers)),
                0.0,
                1.0,
            )
        )
        atk_frac_cli = float(
            np.clip(
                float(sampled_attacker_count) / max(1.0, float(sampled_client_count)),
                0.0,
                1.0,
            )
        )
        bd_success = float(np.clip(float(local_bd_success), 0.0, 1.0))
        delta_lognorm = float(np.log1p(float(np.linalg.norm(delta_full))))

        action = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if action.size < self.config.action_dim:
            padded = np.zeros(self.config.action_dim, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        action = np.clip(action[: self.config.action_dim], -1.0, 1.0).astype(np.float32)

        step_obs = np.concatenate(
            [
                np.asarray(
                    [round_phase, atk_frac_atk, atk_frac_cli, bd_success, delta_lognorm],
                    dtype=np.float32,
                ),
                self.projector.project(_unit_direction(current_tail)),
                self.projector.project(_unit_direction(delta_tail)),
                action,
            ],
            axis=0,
        ).astype(np.float32)

        self.history.append(step_obs)
        self.history = self.history[-max(1, int(self.config.history_window)) :]
        padded = list(self.history)
        while len(padded) < max(1, int(self.config.history_window)):
            padded.insert(0, np.zeros(self.per_step_dim, dtype=np.float32))
        return np.concatenate(padded, axis=0).astype(np.float32)
