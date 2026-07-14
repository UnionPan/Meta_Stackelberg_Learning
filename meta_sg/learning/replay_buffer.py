"""Replay buffer wrapper backed by Tianshou."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from tianshou.data import Batch, ReplayBuffer as TianshouReplayBuffer


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (s, a, r, s', done) tuples,
    while preserving the small project-local API used by the Meta-SG code.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._buffer = TianshouReplayBuffer(size=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.add(
            Batch(
                obs=np.asarray(obs, dtype=np.float32),
                act=np.asarray(action, dtype=np.float32),
                rew=float(reward),
                terminated=bool(done),
                truncated=False,
                obs_next=np.asarray(next_obs, dtype=np.float32),
                info={},
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch, _ = self._buffer.sample(batch_size)
        return (
            np.asarray(batch.obs, dtype=np.float32),
            np.asarray(batch.act, dtype=np.float32),
            np.asarray(batch.rew, dtype=np.float32).reshape(-1, 1),
            np.asarray(batch.obs_next, dtype=np.float32),
            np.asarray(batch.done, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def save(self, path: str | Path) -> None:
        """Persist the complete circular-buffer state using Tianshou's HDF5 format."""

        self._buffer.save_hdf5(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int,
        obs_dim: int,
        act_dim: int,
    ) -> "ReplayBuffer":
        """Load a buffer only when its persisted invariants match the caller."""

        loaded = TianshouReplayBuffer.load_hdf5(str(path))
        if int(loaded.maxsize) != int(capacity):
            raise ValueError("replay buffer invariant mismatch: capacity")

        if len(loaded):
            valid_indices = loaded.sample_indices(0)
            batch = loaded[valid_indices]
            if np.asarray(batch.obs).shape[-1] != int(obs_dim):
                raise ValueError("replay buffer invariant mismatch: obs_dim")
            if np.asarray(batch.act).shape[-1] != int(act_dim):
                raise ValueError("replay buffer invariant mismatch: act_dim")

        result = cls(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
        result._buffer = loaded
        return result

    @property
    def ready(self) -> bool:
        return len(self) > 0

    @property
    def tianshou_buffer(self) -> TianshouReplayBuffer:
        return self._buffer
