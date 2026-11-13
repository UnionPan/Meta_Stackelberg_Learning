"""Export shared index-based splits for federated training."""

from meta_sg.data.partition import (
    partition_dirichlet_indices,
    partition_iid_indices,
    split_root_indices,
)


__all__ = [
    'partition_dirichlet_indices', 'partition_iid_indices', 'split_root_indices',
]
