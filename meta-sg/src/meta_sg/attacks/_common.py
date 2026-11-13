"""Select active malicious population identities shared by both attack categories."""

from collections.abc import Collection, Sequence

def active_clients(
    selected_clients: Sequence[int],
    malicious_clients: Collection[int],
    round_index: int,
    attack_start: int,
) -> list[int]:
    if round_index < attack_start:
        return []
    return [client for client in selected_clients if client in malicious_clients]
