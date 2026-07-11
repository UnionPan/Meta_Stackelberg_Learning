"""Immutable threat-model records for canonical attack execution."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.types import ClientUpdate


@dataclass(frozen=True)
class AttackCapabilities:
    needs_global_model: bool = False
    needs_local_data: bool = False
    observes_benign_updates: bool = False
    observes_other_client_data: bool = False
    observes_private_diagnostics: bool = False
    uses_oracle_data: bool = False


@dataclass(frozen=True)
class AttackKnowledge:
    allows_global_model: bool = False
    allows_local_data: bool = False
    allows_benign_updates: bool = False
    allows_other_client_data: bool = False
    allows_private_diagnostics: bool = False
    allows_oracle_data: bool = False


@dataclass(frozen=True)
class AttackContext:
    client_id: int
    round_index: int
    global_model: ModelState

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError('client_id must be non-negative')
        if self.round_index < 0:
            raise ValueError('round_index must be non-negative')


@dataclass(frozen=True)
class RoundAttackContext:
    """Sanitized evidence available to a round-level malicious generator."""

    round_index: int
    global_model: ModelState | None
    malicious_client_ids: tuple[int, ...]
    benign_updates: tuple[ClientUpdate, ...]

    def __post_init__(self) -> None:
        malicious_ids = tuple(self.malicious_client_ids)
        benign_updates = tuple(self.benign_updates)
        if self.round_index < 0:
            raise ValueError('round_index must be non-negative')
        if any(
            not isinstance(client_id, Integral) or isinstance(client_id, bool)
            for client_id in malicious_ids
        ):
            raise TypeError('malicious client ids must be non-bool integers')
        if any(client_id < 0 for client_id in malicious_ids):
            raise ValueError('malicious client ids must be non-negative')
        if len(malicious_ids) != len(set(malicious_ids)):
            raise ValueError('malicious client ids contain duplicates')
        if any(update.is_malicious for update in benign_updates):
            raise ValueError('benign reference updates must not be malicious')
        benign_ids = tuple(update.client_id for update in benign_updates)
        if len(benign_ids) != len(set(benign_ids)):
            raise ValueError('benign reference updates contain duplicate clients')
        if set(malicious_ids) & set(benign_ids):
            raise ValueError('benign and malicious client ids must be disjoint')
        object.__setattr__(self, 'malicious_client_ids', tuple(int(value) for value in malicious_ids))
        object.__setattr__(self, 'benign_updates', benign_updates)


_CAPABILITY_KNOWLEDGE_PAIRS = (
    ('needs_global_model', 'allows_global_model'),
    ('needs_local_data', 'allows_local_data'),
    ('observes_benign_updates', 'allows_benign_updates'),
    ('observes_other_client_data', 'allows_other_client_data'),
    ('observes_private_diagnostics', 'allows_private_diagnostics'),
    ('uses_oracle_data', 'allows_oracle_data'),
)


def validate_capabilities(
    capabilities: AttackCapabilities,
    knowledge: AttackKnowledge,
) -> None:
    """Reject any information dependency not allowed by the task threat model."""

    forbidden = [
        capability_name
        for capability_name, knowledge_name in _CAPABILITY_KNOWLEDGE_PAIRS
        if getattr(capabilities, capability_name) and not getattr(knowledge, knowledge_name)
    ]
    if forbidden:
        raise ValueError(f'attack capability is forbidden by threat model: {", ".join(forbidden)}')
