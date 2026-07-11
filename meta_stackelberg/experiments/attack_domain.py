"""Typed pretrained attacker policy domains with defense-origin provenance."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import torch

from meta_stackelberg.agents.td3.agent import TD3Agent, TD3Snapshot


@dataclass(frozen=True)
class AttackTypeDomainSource:
    snapshots: Mapping[str, TD3Snapshot]
    origins: Mapping[str, str]
    protocol: str = 'pretrained-attack-type-domain-v1'

    def __post_init__(self) -> None:
        snapshots = dict(self.snapshots)
        origins = dict(self.origins)
        if not snapshots or set(snapshots) != set(origins):
            raise ValueError('attack snapshots and origins must have identical non-empty labels')
        if any(
            not label or not isinstance(snapshot, TD3Snapshot)
            or snapshot.role != 'attacker'
            for label, snapshot in snapshots.items()
        ):
            raise ValueError('attack domain requires labeled attacker snapshots')
        if any(not origin for origin in origins.values()):
            raise ValueError('attack origins must not be empty')
        object.__setattr__(self, 'snapshots', MappingProxyType(snapshots))
        object.__setattr__(self, 'origins', MappingProxyType(origins))

    @classmethod
    def from_policies(
        cls,
        policies: Mapping[str, TD3Agent],
        *,
        origins: Mapping[str, str],
    ) -> 'AttackTypeDomainSource':
        if set(policies) != set(origins):
            raise ValueError('policy labels and origins must match')
        return cls(
            {label: policy.snapshot() for label, policy in policies.items()},
            origins,
        )


def save_attack_type_domain(
    path: str | Path,
    source: AttackTypeDomainSource,
) -> None:
    if not isinstance(source, AttackTypeDomainSource):
        raise TypeError('source must be AttackTypeDomainSource')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'protocol': source.protocol,
        'snapshots': dict(source.snapshots),
        'origins': dict(source.origins),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_attack_type_domain(path: str | Path) -> AttackTypeDomainSource:
    """Load a trusted-local attack domain artifact."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('unknown attack type domain schema')
    return AttackTypeDomainSource(
        payload.get('snapshots', {}),
        payload.get('origins', {}),
        protocol=str(payload.get('protocol', '')),
    )
