"""Role-local replay buffer for canonical TD3."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from hashlib import sha256
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


@dataclass(frozen=True)
class TD3ReplaySnapshot:
    schema_version: int
    capacity: int
    obs_dim: int
    action_dim: int
    role: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    generations: np.ndarray
    position: int
    size: int
    numpy_rng_state: dict


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
    _INITIAL_STORAGE = 1024

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
        self._allocate_storage(min(capacity, self._INITIAL_STORAGE))
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
        self._ensure_storage(index)
        self._observations[index] = obs
        self._actions[index] = act
        self._rewards[index, 0] = float(reward)
        self._next_observations[index] = next_obs
        self._dones[index, 0] = float(bool(done))
        self._generations[index, 0] = generation
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, *, replace: bool = False) -> TD3Batch:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._size and not replace:
            raise ValueError('batch_size exceeds replay size')
        if self._size == 0:
            raise ValueError('cannot sample an empty replay')
        indices = self._rng.choice(self._size, size=batch_size, replace=replace)
        return TD3Batch(
            self._observations[indices].copy(),
            self._actions[indices].copy(),
            self._rewards[indices].copy(),
            self._next_observations[indices].copy(),
            self._dones[indices].copy(),
            self._generations[indices].copy(),
        )

    def snapshot(self) -> TD3ReplaySnapshot:
        rows = self._size
        return TD3ReplaySnapshot(
            2,
            self.capacity,
            self.obs_dim,
            self.action_dim,
            self.role,
            self._observations[:rows].copy(),
            self._actions[:rows].copy(),
            self._rewards[:rows].copy(),
            self._next_observations[:rows].copy(),
            self._dones[:rows].copy(),
            self._generations[:rows].copy(),
            self._position,
            self._size,
            copy.deepcopy(self._rng.bit_generator.state),
        )

    def restore(self, snapshot: TD3ReplaySnapshot) -> None:
        if (
            not isinstance(snapshot, TD3ReplaySnapshot)
            or snapshot.schema_version not in {1, 2}
        ):
            raise ValueError('unknown TD3 replay snapshot schema')
        identity = (self.capacity, self.obs_dim, self.action_dim, self.role)
        if identity != (
            snapshot.capacity, snapshot.obs_dim, snapshot.action_dim, snapshot.role,
        ):
            raise ValueError('TD3 replay snapshot structure mismatch')
        if not 0 <= snapshot.size <= self.capacity:
            raise ValueError('TD3 replay snapshot size is invalid')
        if snapshot.schema_version == 1:
            valid_rows = self.capacity
        else:
            valid_rows = snapshot.size
        expected_shapes = (
            (snapshot.observations, (valid_rows, self.obs_dim)),
            (snapshot.actions, (valid_rows, self.action_dim)),
            (snapshot.rewards, (valid_rows, 1)),
            (snapshot.next_observations, (valid_rows, self.obs_dim)),
            (snapshot.dones, (valid_rows, 1)),
            (snapshot.generations, (valid_rows, 1)),
        )
        if any(array.shape != shape for array, shape in expected_shapes):
            raise ValueError('TD3 replay snapshot array shape mismatch')
        if snapshot.size < self.capacity and snapshot.position != snapshot.size:
            raise ValueError('non-full replay snapshot cursor must equal its size')
        if snapshot.size == self.capacity and not 0 <= snapshot.position < self.capacity:
            raise ValueError('full replay snapshot cursor is invalid')
        stored_rows = snapshot.size
        self._allocate_storage(max(
            min(self.capacity, self._INITIAL_STORAGE), stored_rows,
        ))
        self._observations[:stored_rows] = snapshot.observations[:stored_rows]
        self._actions[:stored_rows] = snapshot.actions[:stored_rows]
        self._rewards[:stored_rows] = snapshot.rewards[:stored_rows]
        self._next_observations[:stored_rows] = snapshot.next_observations[:stored_rows]
        self._dones[:stored_rows] = snapshot.dones[:stored_rows]
        self._generations[:stored_rows] = snapshot.generations[:stored_rows]
        self._position = snapshot.position
        self._size = snapshot.size
        self._rng.bit_generator.state = copy.deepcopy(snapshot.numpy_rng_state)

    def _allocate_storage(self, rows: int) -> None:
        self._observations = np.zeros((rows, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((rows, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((rows, 1), dtype=np.float32)
        self._next_observations = np.zeros(
            (rows, self.obs_dim), dtype=np.float32,
        )
        self._dones = np.zeros((rows, 1), dtype=np.float32)
        self._generations = np.zeros((rows, 1), dtype=np.int64)

    def _ensure_storage(self, index: int) -> None:
        current = len(self._observations)
        if index < current:
            return
        rows = min(self.capacity, max(index + 1, current * 2))
        old = (
            self._observations,
            self._actions,
            self._rewards,
            self._next_observations,
            self._dones,
            self._generations,
        )
        self._allocate_storage(rows)
        copied = min(self._size, current)
        for target, source in zip((
            self._observations,
            self._actions,
            self._rewards,
            self._next_observations,
            self._dones,
            self._generations,
        ), old):
            target[:copied] = source[:copied]

    def fingerprint(self) -> str:
        digest = sha256()
        snapshot = self.snapshot()
        for field in snapshot.__dataclass_fields__:
            _hash_replay_value(digest, getattr(snapshot, field))
        return digest.hexdigest()


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite shape ({size},)')
    return result.copy()


def _hash_replay_value(digest, value) -> None:
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _hash_replay_value(digest, key)
            _hash_replay_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _hash_replay_value(digest, item)
    else:
        digest.update(repr(value).encode())
