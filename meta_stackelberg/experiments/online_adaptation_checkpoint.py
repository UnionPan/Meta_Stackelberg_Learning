"""Atomic resumable checkpoints for one paper online-adaptation scenario."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from meta_stackelberg.agents.td3.agent import TD3Snapshot
from meta_stackelberg.agents.td3.replay import TD3ReplaySnapshot
from meta_stackelberg.stackelberg.policy_online import PolicyOnlineIterationTrace


@dataclass(frozen=True)
class OnlineAdaptationCheckpoint:
    phase: str
    config_signature: Mapping[str, object]
    completed_iterations: int
    meta_defender_fingerprint: str
    adapted_defender: TD3Snapshot
    attacker_fingerprint: str
    replay: TD3ReplaySnapshot
    support_seeds: tuple[int, ...]
    seed_cursor: int
    replay_serial: int
    iterations: tuple[PolicyOnlineIterationTrace, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.phase not in {'adapting', 'complete'}:
            raise ValueError('unknown online-adaptation checkpoint phase')
        if self.completed_iterations != len(self.iterations):
            raise ValueError('online-adaptation history length mismatch')
        if self.adapted_defender.role != 'defender':
            raise ValueError('online-adaptation policy must be a Defender')
        if not 0 <= self.seed_cursor <= len(self.support_seeds):
            raise ValueError('online-adaptation seed cursor is invalid')
        if self.replay_serial < 0:
            raise ValueError('online-adaptation replay serial is invalid')
        if len(self.support_seeds) != len(set(self.support_seeds)):
            raise ValueError('online-adaptation support seeds must be unique')


def save_online_adaptation_checkpoint(
    path: str | Path,
    checkpoint: OnlineAdaptationCheckpoint,
) -> None:
    if not isinstance(checkpoint, OnlineAdaptationCheckpoint):
        raise TypeError('checkpoint must be OnlineAdaptationCheckpoint')
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


def load_online_adaptation_checkpoint(
    path: str | Path,
) -> OnlineAdaptationCheckpoint:
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(value, OnlineAdaptationCheckpoint):
        raise ValueError('unknown online-adaptation checkpoint schema')
    if value.schema_version != 1:
        raise ValueError('unknown online-adaptation checkpoint schema')
    return value
