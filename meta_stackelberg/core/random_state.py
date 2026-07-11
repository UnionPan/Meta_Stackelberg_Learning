"""Explicit random-number sources with replayable state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - canonical core remains importable without Torch
    torch = None


@dataclass(frozen=True)
class RandomSnapshot:
    python_state: object
    numpy_state: dict[str, Any]
    torch_cpu_state: object | None


class RandomSource:
    """Own Python, NumPy, and optional Torch generators for one execution stream."""

    def __init__(self, seed: int) -> None:
        self._python = random.Random(int(seed))
        self._numpy = np.random.default_rng(int(seed))
        self._torch = None
        if torch is not None:
            self._torch = torch.Generator(device='cpu')
            self._torch.manual_seed(int(seed))

    @property
    def python(self) -> random.Random:
        return self._python

    @property
    def numpy(self) -> np.random.Generator:
        return self._numpy

    @property
    def torch(self):
        if self._torch is None:
            raise RuntimeError('Torch is not installed for this RandomSource')
        return self._torch

    def capture(self) -> RandomSnapshot:
        torch_state = self._torch.get_state().clone() if self._torch is not None else None
        return RandomSnapshot(
            python_state=copy.deepcopy(self._python.getstate()),
            numpy_state=copy.deepcopy(self._numpy.bit_generator.state),
            torch_cpu_state=torch_state,
        )

    def spawn(self) -> 'RandomSource':
        """Advance the parent once and return an independently consumable child stream."""

        seed = int(
            self._numpy.integers(
                0,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )
        return RandomSource(seed)

    def restore(self, snapshot: RandomSnapshot) -> None:
        self._python.setstate(copy.deepcopy(snapshot.python_state))
        self._numpy.bit_generator.state = copy.deepcopy(snapshot.numpy_state)
        if snapshot.torch_cpu_state is not None:
            if self._torch is None:
                raise RuntimeError('Cannot restore Torch state because Torch is not installed')
            self._torch.set_state(snapshot.torch_cpu_state.clone())
