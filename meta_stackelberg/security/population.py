"""Fixed malicious-client populations for controlled attack tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, init=False)
class FixedMaliciousPopulation:
    client_ids: frozenset[int]

    def __init__(self, client_ids: Iterable[int]) -> None:
        values = tuple(client_ids)
        if any(not isinstance(client_id, Integral) or isinstance(client_id, bool) for client_id in values):
            raise TypeError('malicious client ids must be integers')
        normalized = frozenset(int(client_id) for client_id in values)
        if any(client_id < 0 for client_id in normalized):
            raise ValueError('malicious client ids must be non-negative')
        object.__setattr__(self, 'client_ids', normalized)

    def contains(self, client_id: int) -> bool:
        if not isinstance(client_id, Integral) or isinstance(client_id, bool):
            return False
        return int(client_id) in self.client_ids
