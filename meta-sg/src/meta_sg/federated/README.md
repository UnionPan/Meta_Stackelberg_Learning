# Federated training

## Client training

[client.py](client.py) trains a supplied model for a fixed number of SGD steps or full DataLoader passes using cross-entropy loss.
It receives normalized batches from a DataLoader, so the training function does not depend on dataset names, image dimensions, or class counts.

```text
Global parameters -> worker model -> client batches -> SGD steps -> parameter delta
```

Call `train_local_update(model, base_state, loader, steps=..., learning_rate=..., device=...)`.
The function loads `base_state`, switches the worker to training mode, and creates a fresh SGD optimizer for each call. Optional `momentum` and `weight_decay` default to zero; optimizer state is never shared between clients or rounds.
It modifies the supplied worker model and leaves `base_state` unchanged.
When the loader is exhausted before the requested steps, a new pass starts.
Passing `epochs=N` instead consumes `N` complete loader passes and takes priority over `steps`. The [JSON runner](../experiments/README.md) constructs shuffled loaders without replacement for epoch mode; the last partial batch is retained.

The return value is `(update, metrics)`:

| Field | Meaning |
| --- | --- |
| `update` | Local parameters minus global parameters, in the worker's parameter order and on its device |
| `metrics['loss']` | Sample-weighted mean of batch losses observed before each optimizer update; `None` for zero steps |
| `metrics['examples_seen']` | Processed examples, including repeated samples |
| `metrics['steps']` | Actual number of SGD steps, including all batches in epoch mode |

### Batch sampling

The original experiment draws a fresh batch **with replacement** at every step.
Configure the loader with `RandomSampler` to preserve that behavior:

```python
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from meta_sg.datasets.mnist import MNISTCNN, load_mnist
from meta_sg.federated import model_state, train_local_update

train, test = load_mnist('data', download=True)
client = Subset(train, list(range(100)))
steps, batch_size = 5, 32
sampler = RandomSampler(
    client,
    replacement=True,
    num_samples=steps * batch_size,
    generator=torch.Generator().manual_seed(42),
)
loader = DataLoader(client, batch_size=batch_size, sampler=sampler)
worker = MNISTCNN()
base_state = model_state(worker)
update, metrics = train_local_update(
    worker, base_state, loader, steps=steps, learning_rate=0.05, device='cpu',
)
```

In a federated round, use a client's partition indices to construct the `Subset` and the current global parameters as `base_state`.
Supplying a shuffled sequential loader instead gives shuffled passes without replacement within each pass.

## Server rounds

[server.py](server.py) connects client training, a supplied defense function, and evaluation.

```text
Sample participants -> train each from the same global parameters -> aggregate
client deltas -> apply the aggregate once -> evaluate the global model
```

`run_federated()` is an iterator. Consuming it first yields a round-zero evaluation, then updates the supplied global model in place and yields one record after each round.
One separate worker model is reused for local training. Parameters and model buffers have separate snapshots; buffers remain outside defended parameter tuples.
`attack=...` can replace the clean client-update collection with an [attack round function](../attacks/README.md).
It returns updates keyed by population ID; the server restores sampled order before passing them to the defense.

Continuing the client-loader example above:

```python
from meta_sg.defenses import fedavg
from meta_sg.federated import run_federated

model = MNISTCNN()
client_loaders = [loader]  # Use one loader per client partition in a full run.
test_loader = DataLoader(test, batch_size=256, shuffle=False)

for record in run_federated(
    model, client_loaders, test_loader, fedavg,
    rounds=3, clients_per_round=1, local_steps=steps,
    learning_rate=0.05, device='cpu', seed=42,
):
    print(record)
```

The server's local NumPy generator samples participants uniformly without replacement within each round.
Optional `local_epochs` selects full passes for client training. Root training always uses `root_steps`.
`learning_rate_decay` defaults to 1 and sets round `r` to `learning_rate * learning_rate_decay ** (r - 1)` for both clients and root. Optional momentum and weight decay are shared by ordinary and backdoor SGD; the frozen RL proxy retains its separate archive settings.
Each loader's sampler controls its own batch sequence across calls; reconstruct or reset those samplers for matched comparisons.
For defense-specific parameters, pass a function bound with `functools.partial`, such as `partial(trimmed_mean, byzantine_bound=1)`.

FLTrust additionally needs `root_loader` and `root_steps`. Each round, the worker trains the root update from the same global snapshot and passes it as `root_update` to the aggregation function.
Leave `root_loader` unset for methods that do not use a trusted root update.

## BatchNorm buffers

[buffers.py](buffers.py) snapshots and restores buffers by name. The server resets each local model's buffers to the same global snapshot before training, including root training.
After local training, floating buffers (BatchNorm running mean and variance) are averaged equally across all sampled clients. Integer counters use their maximum, preserving integer dtype when clients execute different numbers of steps.
With equal local step counts, clean FedAvg therefore averages the full floating model state as in the historical ResNet runners; counters are equal across clients.

Selected defenses aggregate parameter deltas only. Buffer averaging uses all sampled submissions regardless of a defense's selection or trust weights; it provides no robust filtering of BatchNorm statistics.
DBA/BFL submit their locally trained buffers without model-replacement scaling. Synthetic IPM/LMP/Min-Max submissions retain the global buffer snapshot. Root buffers are excluded from the global average.
Round diagnostics record `buffer_aggregation="mean_float_max_counter"` when buffers are present.

For standalone `train_local_update()`, pass `base_buffers=model_buffers(global_model)` to restore buffers along with parameters. Custom attack callbacks for buffered models receive `base_buffers` and a `client_buffers` output dictionary; they must provide one buffer state for every selected client.

| Record field | Contents |
| --- | --- |
| `round` | Zero for the initial evaluation, then one-based training rounds |
| `learning_rate` | Current honest/root SGD rate; initial configured rate for round zero |
| `selected_clients` | Client IDs in the order submitted to aggregation |
| `client_metrics` | Selected client IDs and their local training metrics |
| `root_metrics` | Root training metrics, or `None` |
| `aggregation` | Diagnostics returned by the defense |
| `evaluation` | Global model loss, accuracy, and processed example count |
| `attack` | Construction diagnostics and active malicious identities, or an empty dictionary |
| `backdoor_evaluation` | Optional ASR results, or `None` |

Defense diagnostics such as `selected_indices` refer to positions in `selected_clients`.
The [JSON entry point](../experiments/README.md#saving-and-resuming) prints and persists these records and owns checkpoint/recovery files.
For resumed iteration, `start_round` is the last saved integer round; even round zero skips initial evaluation. Omission starts a fresh run.
An explicitly supplied `participant_generator` lets the entry point save/restore participant sampling state alongside DataLoader generators.

## Learnable aggregation

Optional `defender` supplies [RLDefender](../rl/README.md) instead of a fixed `aggregate` function.
It sees the global model, submitted updates, root update, and trusted-root loader; it does not receive attack diagnostics, malicious IDs, or test labels.
After applying the parameter update and ordinary buffer aggregation, the server supplies root feedback and the episode-terminal flag.
Observation, action, and reward components live outside the server and may be replaced independently. The action component defines the policy dimension; the server does not depend on action names or vector width.

## Evaluation

[evaluation.py](evaluation.py) switches the supplied model to evaluation mode and computes loss and accuracy without gradients in one pass.
Loss is averaged over actual examples, so a smaller final batch has the correct weight. Accuracy is a fraction in `[0, 1]`.
Evaluation leaves the model in evaluation mode; local training switches the worker back to training mode.

## Parameter state and aggregation

[state.py](state.py) snapshots and restores parameter tuples and computes deltas.
Parameter deltas retain the existing parameter-only protocol; [buffers.py](buffers.py) synchronizes buffers separately.

Defense methods consume the client deltas and return an aggregated delta, which `apply_model_update()` adds directly to the global model.
[aggregation.py](aggregation.py) provides a separate weighted-update operation for callers that already have client weights.
