"""Atomic trusted-local round-boundary checkpoints for attack pre-training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import torch

from meta_stackelberg.agents.td3.agent import TD3Snapshot, TD3UpdateStats
from meta_stackelberg.agents.td3.replay import TD3ReplaySnapshot
from meta_stackelberg.federated.types import RoundState


@dataclass(frozen=True)
class AttackPretrainingCheckpoint:
    label: str
    origin: str
    config: Mapping[str, object]
    aggregator_spec: Mapping[str, object]
    round_state: RoundState
    observed_max_norm: float
    last_epsilon: float | None
    pending_observation: np.ndarray
    pending_action: np.ndarray
    pending_reward: float
    attacker_snapshot: TD3Snapshot
    replay_snapshot: TD3ReplaySnapshot
    transition_count: int
    update_stats: tuple[TD3UpdateStats, ...]
    defender_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError('unknown attack pre-training checkpoint schema')
        if not self.label or not self.origin:
            raise ValueError('checkpoint label and origin must not be empty')
        if self.round_state.round_index <= 0:
            raise ValueError('checkpoint must be captured after a completed round')
        if self.transition_count < 0:
            raise ValueError('checkpoint transition_count must be non-negative')
        object.__setattr__(self, 'config', copy.deepcopy(dict(self.config)))
        object.__setattr__(
            self, 'aggregator_spec',
            copy.deepcopy(dict(self.aggregator_spec)),
        )
        object.__setattr__(
            self, 'pending_observation',
            np.asarray(self.pending_observation, dtype=np.float32).copy(),
        )
        object.__setattr__(
            self, 'pending_action',
            np.asarray(self.pending_action, dtype=np.float32).copy(),
        )


def save_attack_pretraining_checkpoint(
    path: str | Path,
    checkpoint: AttackPretrainingCheckpoint,
) -> None:
    if not isinstance(checkpoint, AttackPretrainingCheckpoint):
        raise TypeError('checkpoint must be AttackPretrainingCheckpoint')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent,
    )
    os.close(descriptor)
    try:
        payload = {
            'schema_version': checkpoint.schema_version,
            'label': checkpoint.label,
            'origin': checkpoint.origin,
            'config': dict(checkpoint.config),
            'aggregator_spec': dict(checkpoint.aggregator_spec),
            'round_state': {
                'round_index': checkpoint.round_state.round_index,
                'global_model': checkpoint.round_state.global_model,
                'random_snapshot': checkpoint.round_state.random_snapshot,
                'component_states': dict(checkpoint.round_state.component_states),
            },
            'observed_max_norm': checkpoint.observed_max_norm,
            'last_epsilon': checkpoint.last_epsilon,
            'pending_observation': checkpoint.pending_observation,
            'pending_action': checkpoint.pending_action,
            'pending_reward': checkpoint.pending_reward,
            'attacker_snapshot': checkpoint.attacker_snapshot,
            'replay_snapshot': checkpoint.replay_snapshot,
            'transition_count': checkpoint.transition_count,
            'update_stats': checkpoint.update_stats,
            'defender_fingerprint': checkpoint.defender_fingerprint,
        }
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_attack_pretraining_checkpoint(
    path: str | Path,
) -> AttackPretrainingCheckpoint:
    """Load a checkpoint produced locally; never use untrusted pickle input."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('invalid attack pre-training checkpoint payload')
    round_payload = payload.get('round_state')
    if not isinstance(round_payload, dict):
        raise ValueError('checkpoint round state payload is invalid')
    round_state = RoundState(
        round_index=round_payload['round_index'],
        global_model=round_payload['global_model'],
        random_snapshot=round_payload['random_snapshot'],
        component_states=round_payload.get('component_states', {}),
    )
    return AttackPretrainingCheckpoint(
        label=payload['label'],
        origin=payload['origin'],
        config=payload['config'],
        aggregator_spec=payload['aggregator_spec'],
        round_state=round_state,
        observed_max_norm=payload['observed_max_norm'],
        last_epsilon=payload['last_epsilon'],
        pending_observation=payload['pending_observation'],
        pending_action=payload['pending_action'],
        pending_reward=payload['pending_reward'],
        attacker_snapshot=payload['attacker_snapshot'],
        replay_snapshot=payload['replay_snapshot'],
        transition_count=payload['transition_count'],
        update_stats=tuple(payload['update_stats']),
        defender_fingerprint=payload['defender_fingerprint'],
    )
