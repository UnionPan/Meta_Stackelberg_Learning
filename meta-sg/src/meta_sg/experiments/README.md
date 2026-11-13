# Configuration and entry point

[config.py](config.py) loads JSON into ordinary dictionaries with section defaults.
[run.py](run.py) binds dataset, partition, attack, and defense functions and consumes the existing federated iterator.

From the project directory, after installing the package:

```bash
meta-sg --config configs/mnist.json --dry-run
meta-sg --config configs/mnist.json
```

The equivalent module command is `python -m meta_sg.experiments.run --config configs/mnist.json`.
`--dry-run` resolves names and prints merged settings without reading datasets, loading an RL policy, or starting training. It does not verify data availability or every algorithm parameter.
Run training on the development machine. The examples select CUDA; the model examples use short runs, while the separate BFL example specifies a 200-round experiment.
All paths, including `data_root` and RL checkpoint paths, are relative to the working directory.

## JSON sections

| Section | Fields |
| --- | --- |
| `dataset` | `name`, `data_root`, `options` passed to the selected loader |
| `model` | `name`: `cnn` (default) or `resnet18` for either RGB dataset |
| `partition` | `method`: `iid` or `dirichlet`; `alpha`; `root_samples_per_class` |
| `federation` | `num_clients`, `clients_per_round` |
| `training` | `rounds`, `local_steps`, `local_epochs`, `root_steps`, `batch_size`, `learning_rate`, `learning_rate_decay`, `momentum`, `weight_decay`, `seed`, `device`, `num_workers` |
| `defense` | `name`, `parameters` passed to its aggregation function |
| `attack` | `name`, `malicious_clients`, `start_round`, `parameters` |
| `evaluation` | `batch_size` for clean and optional ASR evaluation |
| `output` | `directory` (null for stdout only), `checkpoint_interval=50` |
| `rl` | `mode`, `checkpoint`, `fixed_action`, `observation`, `action`, `reward`, `agent`; used by `defense.name="rl"` |
| `meta` | Task list and inner/response/outer budgets for the separate `meta-sg-meta` entry; see below |

Omitted fields use [DEFAULTS](config.py). Unknown section fields raise an error so spelling mistakes are visible.
Dataset options and method parameters are passed directly to their functions; there is no second configuration schema.
`local_epochs=null` (default) uses `local_steps` SGD updates and replacement sampling.
Setting `local_epochs` selects that many complete shuffled passes without replacement within each pass, including the last partial batch. It takes priority over `local_steps`; set `local_steps=null` to make an epoch configuration explicit.
`root_steps` remains a fixed-step budget regardless of client epoch settings.

SGD uses `momentum=0` and `weight_decay=0` by default. The optimizer is recreated for each client and root call, so momentum is local to that call.
At round `r >= 1`, the actual learning rate is `learning_rate * learning_rate_decay ** (r - 1)`. The default decay is 1 (constant rate).
Honest, root, and backdoor training share these optimizer settings. A backdoor's optional `malicious_learning_rate` is a constant override; the frozen RL proxy has its separate `attacker_learning_rate`.
Each round record includes the actual honest/root `learning_rate`; round zero reports the configured initial value.

## Dataset selection

| Name | Loader options | Model |
| --- | --- | --- |
| `mnist` | `download` (default false) | `MNISTCNN` |
| `cifar10` | `download`, `augment` (both default false) | `CIFAR10CNN` or CIFAR-stem ResNet-18 |
| `tiny_imagenet` | `train_per_class=500`, `val_per_class=50`, `seed=42`, `normalization="legacy"`, `augment=false` | `TinyImageNetCNN` or standard ResNet-18 |

Tiny ImageNet's loader seed controls per-class image selection independently of `training.seed`.
Its labeled validation split supplies evaluation data; the unlabeled test split is unused.
Use `"model": {"name": "resnet18"}` with either RGB dataset; MNIST supports `cnn` only.
CIFAR ResNet-18 uses a 3×3 stride-one stem without maxpool; Tiny ResNet-18 retains the standard 7×7 stride-two stem and maxpool. Both retain BatchNorm and use no pretrained weights.
Model selection does not change preprocessing. The [CIFAR ResNet example](../../../configs/cifar10_resnet18.json) keeps 0.5/0.5 normalization without augmentation; the [Tiny ResNet example](../../../configs/tiny_imagenet_resnet18.json) selects `tiny` normalization and augmentation explicitly.
The ResNet model examples retain short fixed-step runs. The [Tiny BFL example](../../../configs/tiny_imagenet_bfl.json) demonstrates epoch training, decay, and weight decay through configuration.

Root examples are sampled first and excluded from every client's training partition for all defenses.
Set `root_samples_per_class` to zero when no holdout is desired. FLTrust requires a positive root sample count and positive `root_steps`.
Only FLTrust trains on the held-out root set; its transform matches deterministic evaluation preprocessing even when client augmentation is enabled.
IID shuffles and divides the client pool; it does not use `alpha`. Dirichlet uses `alpha` to control per-class client proportions.
Client loaders use separately seeded generators. Fixed-step mode samples with replacement; epoch mode shuffles each complete pass. Root batches always use replacement sampling.
For models with buffers, the entry point uses the [shared buffer protocol](../federated/README.md#batchnorm-buffers), which is also recorded in round aggregation diagnostics.

## Defense selection

Supported fixed names: `fedavg`, `median`, `trimmed_mean`, `krum`, `multi_krum`, `rfa`, `fltrust`, `clipmed`.
`rl` selects the [learnable mechanism and RL Defender](../rl/README.md), with independently configurable observation, action, and reward components. It also requires a positive trusted-root holdout.
`rl.action.name` selects mixed norm, median clipping, trusted normalization, or four-control CNB; `controls`/`fixed` configure learned and fixed controls, and `parameters` configures the actuator. The selected names determine policy width.
For example:

```json
"defense": {"name": "clipmed", "parameters": {"norm_bound": 0.1}}
```

Algorithm parameters must match the sampled population: Krum requires `clients_per_round >= 2 * byzantine_bound + 3`; Trimmed Mean requires `clients_per_round > 2 * byzantine_bound`; Multi-Krum also requires `selected <= clients_per_round`.
The defaults (`byzantine_bound=4`, Multi-Krum `selected=16`) must be adjusted when using the example's ten participants.

## Attack selection

`malicious_clients` lists fixed, zero-based IDs in the population. `start_round` is one-based; earlier rounds are clean.
Supported names: `none`, `ipm`, `lmp`, `min_max`, `dba`, `bfl`, `rl`.

```json
"attack": {
  "name": "ipm", "malicious_clients": [0, 1, 2], "start_round": 3,
  "parameters": {"scaling_factor": 5.0}
}
```

| Attack | Method parameters |
| --- | --- |
| IPM | `scaling_factor` |
| LMP | `oracle_bound`, `search_steps`, `maximum_multiplier` |
| Min-Max | `direction` (`std`, `sign`, `unit_vec`), `initial_lambda`, `tolerance` |
| DBA | `target_label`, `source_label`, `poison_fraction`, `poison_selection`, `model_replacement_scale`, `scale_by_malicious_count`, `cyclic`, `malicious_steps`, `malicious_learning_rate` |
| BFL | The same poisoning, scaling, and training parameters; uses the complete trigger |
| Frozen RL | `checkpoint`, `known_indices`, `known_normalization`, `attacker_learning_rate` |

Model-poisoning requires selected honest reference updates; Min-Max needs at least two. LMP's Krum oracle also needs a valid population for its bound. Random participation can produce rounds that do not satisfy these conditions; those rounds report an error.

DBA defaults to all non-target sources, fraction 0.5, and replacement scale 10. BFL defaults to source 0 on MNIST/CIFAR-10 and all non-target sources on Tiny ImageNet.
BFL defaults to fraction 1 and scale 1; both can be overridden through `attack.parameters`.
The default target is 2 for MNIST and 9 for the RGB datasets. Set source and target explicitly to define a different attack task.
`poison_selection="random"` (default) chooses eligible examples with the seeded poisoning generator; `"first"` takes the first eligible examples in batch order.
The count is capped at `round(actual_batch_size * poison_fraction)`, excluding original target examples and applying the optional source filter.
`scale_by_malicious_count=true` divides `model_replacement_scale` by the number of active sampled malicious clients. With no active attackers, no scaling is applied and no division occurs. The default is false (constant scale).
`malicious_steps` overrides the honest training budget, including epoch mode; omission inherits the honest step/epoch budget. Batch sampling still follows the configured client-loader mode.
Triggers and normalization follow the selected dataset. ASR uses the complete trigger on held-out eligible examples and excludes original target examples.
These shared training mechanics do not reproduce every historical dataset-specific attack script; see the [attack module](../attacks/README.md).

The [Tiny BFL configuration](../../../configs/tiny_imagenet_bfl.json) uses standard ResNet-18, five local epochs, SGD rate `0.1 * 0.998 ** (round - 1)`, weight decay 0.001, all non-target sources to target 9, fraction 0.3125, first-eligible poisoning, and scale `10 / active_attackers` from round 101.
Its explicit malicious identities `0..19` are example inputs, not the historical randomly selected identities. Shared partition, batch, and augmentation RNG streams also differ from the old scripts; this preset expresses the settings without claiming identical historical trajectories.

The frozen Li2022 RL attacker is MNIST-only and needs the optional `rl` dependency plus an external checkpoint:

```json
"attack": {
  "name": "rl", "malicious_clients": [0, 1], "start_round": 3,
  "parameters": {
    "checkpoint": "/path/to/final_policy.zip",
    "known_indices": [0, 1, 2, 3],
    "known_normalization": "defender",
    "attacker_learning_rate": 0.01
  }
}
```

`known_indices` selects attacker-known samples from the training dataset, never the evaluation split.
`defender` uses ordinary dataset preprocessing; `archive` uses the historical replacement normalization.
The attacker learning rate is independent of `training.learning_rate`. Policy loading preserves the global model's seeded initialization and CPU RNG stream.

## Saving and resuming

The example JSON files enable `output.directory`. `--output outputs/my_run` overrides that directory for a fresh run, which requires an empty directory.
`output.directory=null` keeps the programmatic/stdout-only mode without filesystem artifacts.

| File | Contents |
| --- | --- |
| `config.json` | Merged settings, resolved input/output paths, and TD3 defaults |
| `partitions.pt` | Actual client and trusted-root training indices |
| `metadata.json` | Dataset image/label fingerprints, partition fingerprint, runtime versions, and any frozen-policy fingerprint |
| `metrics.jsonl` | One flushed JSON record per completed round, including round zero |
| `latest.pt` | Most recent complete-round model, BatchNorm buffers, round, participant RNG, loader/sampler RNGs, Python/NumPy/Torch RNGs, device RNG, and optional RL training state |
| `status.json` | Running/complete/interrupted/failed state, latest completed round, checkpoint round, target budget, and error when applicable |
| `policy.zip`, `policy.json` | Separate inference export for a trained RL Defender |
| `observation.json` | Frozen Histogram40 transform; also embedded in the checkpoint and policy export |

Checkpoints are written at round zero, every `checkpoint_interval` rounds, and at the final round. Only `latest.pt` is retained; writes replace it atomically using a temporary file. JSON status/config files use the same replacement pattern.
Client SGD momentum is local to each call, so there is no cross-round SGD optimizer state to save. The learning-rate schedule derives from the absolute round. TD3 optimizer, replay, target, RNG, and pending-transition state are persisted.

On the development machine:

```bash
meta-sg --config configs/mnist.json --output outputs/mnist_run
meta-sg --resume outputs/mnist_run/latest.pt
```

Resume loads the saved configuration when `--config` is omitted. To extend the total budget, increase `training.rounds` in a configuration and pass both `--config` and `--resume`.
It continues in the checkpoint's existing directory. Experiment settings must match; the total round budget and checkpoint interval may change, and input paths may relocate if their content fingerprints match.
Saved partitions are loaded directly. Metrics after the checkpoint offset are discarded, including any incomplete final line, before recomputing those rounds. Earlier records are retained without repeating initial evaluation.
Round-internal interruption restarts from the last saved round; no partial-client resume is provided.
Replay equivalence assumes unchanged code/runtime/device and deterministic operations; CUDA/MPS backend nondeterminism is not removed by checkpointing.

The entry point prints one JSON object per line, beginning with round-zero evaluation.
Records contain clean accuracy/loss, client and root training metrics, aggregation diagnostics, attack diagnostics, and optional ASR.
On resume, stdout contains only newly executed rounds. `--dry-run` prints settings without creating an output directory or loading datasets/policies; with `--resume` it reads the checkpoint's saved settings.

Histogram40 uses a training-fitted transform file. `--fit-observation histogram40` with `--calibration` and `--output` fits it from training-run metrics JSONL or raw feature NPZ, without running FL/RL. See [Histogram40 fitting and configuration](../rl/DISTRIBUTIONS.md).

## Meta-SG entry

`meta-sg-meta --config configs/mnist_meta.json --dry-run` inspects the dedicated
meta-training configuration. Remove `--dry-run` only on the development machine
to run it; `--resume outputs/mnist_meta/latest.pt` restores the last saved outer
boundary. Ordinary `meta-sg` keeps the existing single-policy behavior.

The `meta` section selects `tasks`, `iterations`, `tasks_per_iteration`, `horizon`,
`inner_steps`, `response_steps`, `post_critic_steps`, `outer_learning_rate`,
`follower_agent`, `feedback_samples`, and the outer `checkpoint_interval`.
Task kinds are `model_poisoning` and `backdoor`; each task has a unique name,
malicious client IDs and its own persistent follower. See
[algorithm, follower parameters and artifacts](../rl/META.md).
