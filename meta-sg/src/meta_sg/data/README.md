# Data processing

Dataset-specific loading, models, and documentation live under `datasets/<name>/`.
This module provides shared partition functions that operate on training indices.

## Output convention

| Level | Output |
| --- | --- |
| Dataset loader | `(train_dataset, test_dataset)` |
| Dataset sample | `(image, label)`; `float32 [C, H, W]` image and integer label |
| DataLoader batch | `float32 [B, C, H, W]` images and `int64 [B]` labels |
| Root split | `(root_indices, remaining_indices)` |
| Client split | A list of index tensors, one per client |

Channels, image dimensions, and preprocessing belong to each dataset. Data remains on the CPU until the training loop transfers a batch to the model's device.

## Example: MNIST

The [MNIST package](../datasets/mnist/README.md) defines loading and preprocessing.
Downloading is opt-in via `download=True`. Local data is read from `data_root/MNIST/`.

```python
from torch.utils.data import DataLoader, Subset

from meta_sg.data import partition_iid_indices, split_root_indices
from meta_sg.datasets.mnist import load_mnist

train, test = load_mnist('data', download=True)
root_indices, remaining = split_root_indices(train.targets, samples_per_class=20, seed=42)
client_indices = partition_iid_indices(remaining, num_clients=100, seed=42)

root = Subset(train, root_indices.tolist())
client = Subset(train, client_indices[0].tolist())
client_loader = DataLoader(client, batch_size=32, shuffle=True)
root_loader = DataLoader(root, batch_size=64, shuffle=False)
test_loader = DataLoader(test, batch_size=256, shuffle=False)
```

## Partitioning

[partition.py](partition.py) first samples the same number of trusted examples from each observed class, then excludes those examples from the client pool.
IID partitioning shuffles the remaining indices and divides them into client subsets whose sizes differ by at most one.
Every remaining example belongs to exactly one client. The official test split is kept separate.

### Dirichlet non-IID

To replace IID partitioning in the example above:

```python
from meta_sg.data import partition_dirichlet_indices

client_indices = partition_dirichlet_indices(
    train.targets, remaining, num_clients=100, alpha=0.5, seed=42,
)
```

For each class, the function draws client proportions from a symmetric Dirichlet distribution with concentration `alpha`, then samples client counts from a multinomial distribution.
Smaller positive `alpha` usually produces stronger label skew; larger values make client label distributions more similar.
Client sample counts can differ. The input indices should be the remaining training indices after root sampling.

The function retries the full partition when any client is empty, up to 100 attempts, and reports failure if no nonempty partition is found.
It returns the same list-of-index-tensors format as IID partitioning.

All partition functions use local seeded generators and preserve the old experiment's sampling order.
The old Dirichlet argument named `concentration` is exposed as `alpha` in this module.
The caller supplies valid sample counts and client counts. The functions report insufficient per-class root samples or insufficient data for nonempty clients.
