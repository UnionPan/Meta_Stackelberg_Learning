"""Atomic trusted-local checkpoints for complete TD3 training state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import torch

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer, TD3ReplaySnapshot


@dataclass(frozen=True)
class TD3TrainingCheckpoint:
    agent_snapshot: TD3Snapshot
    replay_snapshot: TD3ReplaySnapshot
    agent_fingerprint: str
    replay_fingerprint: str
    obs_dim: int
    action_dim: int
    metadata: Mapping[str, object]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'metadata', MappingProxyType(copy.deepcopy(dict(self.metadata))),
        )


def save_td3_training_checkpoint(
    path: str | Path,
    *,
    agent: TD3Agent,
    replay: TD3ReplayBuffer,
    metadata: Mapping[str, object],
) -> TD3TrainingCheckpoint:
    if replay.role != agent.role:
        raise ValueError('checkpoint agent and replay roles must match')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = TD3TrainingCheckpoint(
        agent_snapshot=agent.snapshot(),
        replay_snapshot=replay.snapshot(),
        agent_fingerprint=agent.fingerprint(),
        replay_fingerprint=replay.fingerprint(),
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        metadata=metadata,
    )
    payload = {
        'schema_version': checkpoint.schema_version,
        'agent_snapshot': checkpoint.agent_snapshot,
        'replay_snapshot': checkpoint.replay_snapshot,
        'agent_fingerprint': checkpoint.agent_fingerprint,
        'replay_fingerprint': checkpoint.replay_fingerprint,
        'obs_dim': checkpoint.obs_dim,
        'action_dim': checkpoint.action_dim,
        'metadata': dict(checkpoint.metadata),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return checkpoint


def load_td3_training_checkpoint(
    path: str | Path,
    *,
    agent: TD3Agent,
    replay: TD3ReplayBuffer,
) -> TD3TrainingCheckpoint:
    """Load a checkpoint created locally by this package.

    PyTorch pickle checkpoints must never be loaded from an untrusted source.
    """
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('unknown TD3 training checkpoint schema')
    agent_snapshot = payload.get('agent_snapshot')
    replay_snapshot = payload.get('replay_snapshot')
    if not isinstance(agent_snapshot, TD3Snapshot) or not isinstance(
        replay_snapshot, TD3ReplaySnapshot,
    ):
        raise ValueError('TD3 training checkpoint payload is invalid')
    if agent_snapshot.role != agent.role or replay_snapshot.role != replay.role:
        raise ValueError('TD3 training checkpoint role mismatch')
    if (payload.get('obs_dim'), payload.get('action_dim')) != (
        agent.obs_dim, agent.action_dim,
    ):
        raise ValueError('TD3 training checkpoint policy shape mismatch')
    replay_identity = (
        replay_snapshot.capacity,
        replay_snapshot.obs_dim,
        replay_snapshot.action_dim,
    )
    if replay_identity != (replay.capacity, replay.obs_dim, replay.action_dim):
        raise ValueError('TD3 training checkpoint replay shape mismatch')
    checkpoint = TD3TrainingCheckpoint(
        agent_snapshot=agent_snapshot,
        replay_snapshot=replay_snapshot,
        agent_fingerprint=str(payload.get('agent_fingerprint')),
        replay_fingerprint=str(payload.get('replay_fingerprint')),
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        metadata=payload.get('metadata', {}),
    )
    agent.restore(checkpoint.agent_snapshot)
    replay.restore(checkpoint.replay_snapshot)
    if agent.fingerprint() != checkpoint.agent_fingerprint:
        raise ValueError('restored TD3 agent fingerprint does not match checkpoint')
    if replay.fingerprint() != checkpoint.replay_fingerprint:
        raise ValueError('restored TD3 replay fingerprint does not match checkpoint')
    return checkpoint
