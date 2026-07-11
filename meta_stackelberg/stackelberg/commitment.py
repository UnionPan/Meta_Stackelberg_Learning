"""Versioned immutable defender commitments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from meta_stackelberg.security.defenses.actions import DefenseAction


_AGGREGATION_FAMILY = 'clipped_trimmed_mean'
_EXECUTION_PROTOCOL = 'clip-then-equal-coordinate-trim-v1'
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DefenderCommitment:
    commitment_id: str
    action: DefenseAction
    policy_fingerprint: str
    aggregation_family: str = _AGGREGATION_FAMILY
    execution_protocol: str = _EXECUTION_PROTOCOL
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.commitment_id, str):
            raise TypeError('commitment_id must be a string')
        if not self.commitment_id.strip():
            raise ValueError('commitment_id must be non-empty')
        if not isinstance(self.action, DefenseAction):
            raise TypeError('action must be a DefenseAction')
        if self.aggregation_family != _AGGREGATION_FAMILY:
            raise ValueError('unknown aggregation_family')
        if self.execution_protocol != _EXECUTION_PROTOCOL:
            raise ValueError('unknown execution_protocol')
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError('unknown commitment schema_version')
        if self.policy_fingerprint != self._canonical_fingerprint():
            raise ValueError('policy fingerprint does not match commitment')

    @classmethod
    def create(
        cls,
        commitment_id: str,
        action: DefenseAction,
    ) -> DefenderCommitment:
        provisional = object.__new__(cls)
        object.__setattr__(provisional, 'commitment_id', commitment_id)
        object.__setattr__(provisional, 'action', action)
        object.__setattr__(provisional, 'aggregation_family', _AGGREGATION_FAMILY)
        object.__setattr__(provisional, 'execution_protocol', _EXECUTION_PROTOCOL)
        object.__setattr__(provisional, 'schema_version', _SCHEMA_VERSION)
        object.__setattr__(provisional, 'policy_fingerprint', '')
        fingerprint = provisional._canonical_fingerprint()
        return cls(commitment_id, action, fingerprint)

    def verify(self) -> None:
        if self.policy_fingerprint != self._canonical_fingerprint():
            raise ValueError('policy fingerprint does not match commitment')

    def _canonical_fingerprint(self) -> str:
        if not isinstance(self.action, DefenseAction):
            raise TypeError('action must be a DefenseAction')
        payload = {
            'schema_version': self.schema_version,
            'commitment_id': self.commitment_id,
            'aggregation_family': self.aggregation_family,
            'clip_radius_hex': self.action.clip_radius.hex(),
            'trim_ratio_hex': self.action.trim_ratio.hex(),
            'execution_protocol': self.execution_protocol,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        return sha256(encoded).hexdigest()
