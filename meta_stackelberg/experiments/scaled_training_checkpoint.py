"""Atomic outer-iteration checkpoints for canonical scaled Meta-SG training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from meta_stackelberg.agents.td3.agent import TD3Snapshot
from meta_stackelberg.stackelberg.policy_algorithm1 import (
    PolicyAlgorithm1IterationTrace,
)
from meta_stackelberg.stackelberg.policy_algorithm2 import (
    PolicyAlgorithm2IterationTrace,
)


@dataclass(frozen=True)
class ScaledTrainingCheckpoint:
    phase: str
    config_signature: Mapping[str, object]
    algorithm1_completed: int
    algorithm2_completed: int
    algorithm1_defender: TD3Snapshot
    algorithm2_defender: TD3Snapshot
    algorithm1_attackers: Mapping[object, TD3Snapshot]
    support_seeds: tuple[int, ...]
    next_support_seed: int
    replay_serial: int
    algorithm1_iterations: tuple[PolicyAlgorithm1IterationTrace, ...]
    algorithm2_iterations: tuple[PolicyAlgorithm2IterationTrace, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.phase not in {'algorithm1', 'algorithm2', 'complete'}:
            raise ValueError('unknown scaled training checkpoint phase')
        if self.algorithm1_completed != len(self.algorithm1_iterations):
            raise ValueError('Algorithm 1 checkpoint history length mismatch')
        if self.algorithm2_completed != len(self.algorithm2_iterations):
            raise ValueError('Algorithm 2 checkpoint history length mismatch')
        if self.next_support_seed < 0 or self.replay_serial < 0:
            raise ValueError('checkpoint counters must be non-negative')
        if len(self.support_seeds) != len(set(self.support_seeds)):
            raise ValueError('checkpoint support seeds must be unique')
        if self.algorithm1_defender.role != 'defender':
            raise ValueError('Algorithm 1 checkpoint policy must be a Defender')
        if self.algorithm2_defender.role != 'defender':
            raise ValueError('Algorithm 2 checkpoint policy must be a Defender')
        if not self.algorithm1_attackers or any(
            snapshot.role != 'attacker'
            for snapshot in self.algorithm1_attackers.values()
        ):
            raise ValueError('checkpoint must contain attacker policy snapshots')


def save_scaled_training_checkpoint(
    path: str | Path,
    checkpoint: ScaledTrainingCheckpoint,
) -> None:
    if not isinstance(checkpoint, ScaledTrainingCheckpoint):
        raise TypeError('checkpoint must be ScaledTrainingCheckpoint')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent,
    )
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_scaled_training_checkpoint(
    path: str | Path,
) -> ScaledTrainingCheckpoint:
    """Load a trusted-local canonical training checkpoint."""
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(value, ScaledTrainingCheckpoint):
        raise ValueError('unknown scaled training checkpoint schema')
    if value.schema_version != 1:
        raise ValueError('unknown scaled training checkpoint schema')
    return value
