"""Shared warmup-weight helpers for Stackelberg FL experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from meta_sg.simulation.types import SimulationSnapshot


def clone_weights(weights: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(layer).copy() for layer in weights]


def reset_env_from_weights(env, *, seed: int, weights: list[np.ndarray]):
    env.reset(seed=seed) if hasattr(env, "reset") else env.coordinator.reset(seed=seed)
    start_round_idx = int(getattr(env.coordinator.config.runtime, "start_round_idx", 1) or 1)
    env.coordinator.restore(
        SimulationSnapshot(
            round_idx=start_round_idx - 1,
            weights=clone_weights(weights),
            rng_state=None,
        )
    )
    if hasattr(env, "_round"):
        env._round = 0
    if hasattr(env, "_history"):
        env._history = []
    if hasattr(env, "_make_obs"):
        obs = env._make_obs(env.coordinator.current_weights)
        env._obs = obs
        env._obs_dim = obs.shape[0]
        return obs
    return env.coordinator.current_weights
