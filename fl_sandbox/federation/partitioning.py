"""Client data partitioning strategies."""

from __future__ import annotations

from dataclasses import dataclass
import random

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class DataPartitioner:
    """Assign dataset indices to FL clients."""

    targets: object
    num_clients: int
    seed: int = 42
    noniid_q: float = 0.5

    def targets_list(self) -> list[int]:
        if torch is not None and isinstance(self.targets, torch.Tensor):
            return [int(value) for value in self.targets.cpu().tolist()]
        return [int(value) for value in list(self.targets)]

    def client_groups(self) -> list[int]:
        classes = sorted({int(label) for label in self.targets_list()})
        num_groups = max(1, len(classes))
        return [client_id % num_groups for client_id in range(self.num_clients)]

    def split_iid(self, dataset_size: int) -> list[set[int]]:
        num_items = dataset_size // self.num_clients
        all_indices = list(range(dataset_size))
        client_data_idxs = []
        rng = random.Random(self.seed)
        for _ in range(self.num_clients):
            chosen = set(rng.sample(all_indices, min(num_items, len(all_indices))))
            client_data_idxs.append(chosen)
            all_indices = list(set(all_indices) - chosen)
        return client_data_idxs
