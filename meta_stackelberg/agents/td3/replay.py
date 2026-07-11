"""Role-local replay buffer for canonical TD3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TD3Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    generations: np.ndarray


def flatten_observation(
    observation: Mapping[str, np.ndarray],
    key_order: Sequence[str],
) -> np.ndarray:
    keys = tuple(key_order)
    if not keys or set(keys) != set(observation):
        raise ValueError('observation key order must cover every key exactly')
    parts = []
    for key in keys:
        value = np.asarray(observation[key])
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise ValueError(f'observation field {key!r} must be finite floating data')
        parts.append(value.reshape(-1).astype(np.float32))
    return np.concatenate(parts)


class TD3ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        obs_dim: int,
        action_dim: int,
        role: str,
        seed: int,
    ) -> None:
        for value, name in ((capacity, 'capacity'), (obs_dim, 'obs_dim'), (action_dim, 'action_dim')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if role not in {'defender', 'attacker'}:
            raise ValueError('role must be defender or attacker')
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.role = role
        self._rng = np.random.default_rng(seed)
        self._observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self._dones = np.empty((capacity, 1), dtype=np.float32)
        self._generations = np.empty((capacity, 1), dtype=np.int64)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
        *,
        generation: int,
        role: str,
    ) -> None:
        if role != self.role:
            raise ValueError('transition role does not match replay role')
        obs = _vector(observation, self.obs_dim, 'observation')
        act = _vector(action, self.action_dim, 'action')
        next_obs = _vector(next_observation, self.obs_dim, 'next_observation')
        if isinstance(reward, bool) or not np.isfinite(float(reward)):
            raise ValueError('reward must be finite')
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError('generation must be a non-negative integer')
        index = self._position
        self._observations[index] = obs
        self._actions[index] = act
        self._rewards[index, 0] = float(reward)
        self._next_observations[index] = next_obs
        self._dones[index, 0] = float(bool(done))
        self._generations[index, 0] = generation
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TD3Batch:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._size:
            raise ValueError('batch_size exceeds replay size')
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TD3Batch(
            self._observations[indices].copy(),
            self._actions[indices].copy(),
            self._rewards[indices].copy(),
            self._next_observations[indices].copy(),
            self._dones[indices].copy(),
            self._generations[indices].copy(),
        )


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite shape ({size},)')
    return result.copy()
