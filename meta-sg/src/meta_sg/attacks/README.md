# Attacks

Attacks are grouped by objective: model poisoning degrades the clean task; backdoors induce target predictions on triggered inputs.
The archived untargeted RL attacker belongs to model poisoning.
All submissions use additive parameter deltas: `local parameters - global parameters`.

| Category | Method | Module | Construction and information used |
| --- | --- | --- | --- |
| Model poisoning | IPM | [ipm.py](model_poisoning/ipm.py) | Scaled negative mean of selected honest updates |
| Model poisoning | LMP-Krum | [lmp.py](model_poisoning/lmp.py) | Sign-direction search using selected honest updates and a Krum oracle |
| Model poisoning | Min-Max | [min_max.py](model_poisoning/min_max.py) | Distance-constrained search using selected honest updates |
| Model poisoning | Frozen RL | [rl.py](model_poisoning/rl.py) | Archived Li2022 policy; global classifier parameters, attacker count, and known training data |
| Backdoor | DBA | [dba.py](backdoor/dba.py) | Stable sub-triggers, own client batches, optional cyclic schedule and model replacement |
| Backdoor | BFL | [bfl.py](backdoor/bfl.py) | Complete trigger on eligible examples; configurable poisoning and replacement scale |

## Round injection

Bind an attack with `functools.partial` and pass it as `attack=...` to [run_federated](../federated/server.py).
The server supplies its worker, global parameter snapshot, client loaders, sampled IDs, round number, local training settings, device, and seed.
Training settings include `epochs`, `momentum`, and `weight_decay`; the supplied learning rate already includes the server's round decay. Honest clients use those settings under every attack.
The attack returns `(updates_by_client_id, client_metrics, attack_metrics)`.

Malicious IDs refer to persistent population identities, independently of their positions in a round's sample.
Before `attack_start`, or when no malicious client is sampled, participants train normally.
The server restores sampled order before aggregation. Attack identities and construction diagnostics are recorded separately from the defense's inputs.

## Model poisoning

```python
from functools import partial

from meta_sg.attacks.model_poisoning import ipm_updates, lmp_updates, min_max_updates, model_poisoning_round

attack = partial(
    model_poisoning_round,
    malicious_clients={0, 1},
    craft=partial(ipm_updates, scaling_factor=5),
    attack_start=3,
)
```

Replace `craft` with `partial(lmp_updates, oracle_bound=2)` or `partial(min_max_updates, direction='std')` to select the other methods.
Only honest participants train; malicious submissions are constructed once and copied independently.
Their unused clean local training is skipped, so `client_metrics` contains only clients that actually trained.

LMP's `oracle_bound` defaults to the actual number of selected attackers; an explicit value reproduces a fixed oracle budget with varying participation.
This is the Krum-targeted variant, not the older root-loss-targeted diagnostic variant.
The search reports construction failure if Krum accepts no candidate.
Min-Max needs at least two honest reference updates and follows the existing repository's model-delta adaptation.

### Frozen RL attacker

Install the optional dependency with `uv sync --group dev --extra rl`.
The original checkpoint and known training-example IDs are external experiment inputs.

```python
from functools import partial

from meta_sg.attacks.model_poisoning import FrozenLi2022Actor, rl_round

actor = FrozenLi2022Actor('path/to/final_policy.zip')
attack = partial(
    rl_round, actor=actor, malicious_clients={0, 1},
    known_images=known_images, known_labels=known_labels,
    attacker_learning_rate=0.05, attack_start=3,
)
```

The actor is frozen and requires MNIST's 128-to-10 classifier: 1,290 min-max-scaled parameters plus a selected attacker count from 0 through 10.
Its two actions decode to `gamma=(a0+1)*5` and `steps=round((a1+1)*25)`.
The crafted update is `-gamma * SGD_delta`, using eval-mode SGD with shuffled batches of up to 200 known examples.
`attacker_learning_rate` configures this proxy SGD independently of the server's honest-client/root `learning_rate`.
It defaults to `0.01` for archive compatibility; use `0.05` for the historical higher-rate attack condition.
The configured value is included in each round's `attack` diagnostics. The actor policy remains frozen.
The supplied archive contains 200 known examples; use its manifest's IDs for archive-matched evaluation.

Prepare `known_images` and `known_labels` from available training examples, with the intended preprocessing.
`archived_normalization(raw_training_images)` provides the archive's replacement mean/std; standard defender normalization is a separate transfer condition.
The module loads an existing policy and does not train an RL attacker.
Known-data SGD runs once per attacked round; malicious clients receive independent copies of that update.
Its random draws are isolated from other Torch work with `fork_rng`.
The frozen proxy retains eval-mode, fixed-step SGD without momentum or weight decay; honest-client optimizer settings do not change the archived attack construction.

## Backdoor

```python
from functools import partial

from meta_sg.attacks.backdoor import bfl_round, dba_round
from meta_sg.datasets.mnist import dba_trigger

attack = partial(
    dba_round, malicious_clients={0, 1, 2, 3}, trigger=dba_trigger,
    target_label=2, poison_fraction=0.5, model_replacement_scale=10,
    malicious_steps=10, attack_start=3,
)

bfl_attack = partial(
    bfl_round, malicious_clients={0, 1, 2, 3}, trigger=dba_trigger,
    source_label=0, target_label=2, malicious_steps=10, attack_start=3,
)
```

DBA assigns sub-trigger `rank % 4` using each client's rank in the sorted malicious population.
Set `cyclic=True` to activate one population identity per round; it attacks only if sampled.
BFL uses the complete trigger. Its defaults poison all eligible source samples and return unscaled local updates; `poison_fraction` and `model_replacement_scale` can override those defaults.
`source_label=None` selects all non-target classes. Set an integer to restrict source eligibility.
`malicious_steps` overrides the honest training budget, including epoch mode. Omission inherits the honest step/epoch budget; batch sampling remains controlled by the client loader.
`malicious_learning_rate` optionally sets a constant malicious SGD rate; omission inherits the server's current decayed rate. Momentum and weight decay follow the common training settings.

Set `scale_by_malicious_count=True` to divide the configured replacement scale by the number of active sampled attackers, after warmup and cyclic scheduling. With zero active attackers there is no division or replacement.
Only parameter deltas are scaled; BatchNorm buffers retain their ordinary client statistics.

[training.py](backdoor/training.py) poisons at most `round(batch_size * poison_fraction)` eligible examples in each batch, excluding the target class and applying `source_label` when supplied.
`poison_selection='random'` selects with the local generator; `'first'` takes eligible examples in their batch order. Both use the actual batch size, including a partial final batch.
Dataset-specific triggers consume preprocessed tensors. [The MNIST trigger](../datasets/mnist/triggers.py) writes normalized white pixels matching raw-white poisoning before normalization.

The DataLoader owns batch sampling. Poison selection uses a separate local generator seeded with `seed + round_index * population_size + client_id`.
This differs from the old raw-tensor helper's shared batch/poison generator; multi-step trajectories are not claimed to be bit-identical to that helper.
The [Tiny BFL example](../../../configs/tiny_imagenet_bfl.json) expresses the later BFL settings through the existing `bfl` method: fraction 0.3125, first-eligible poisoning, all non-target sources, target 9, and replacement scale `10 / active_attackers`.

### Attack success rate

[evaluation.py](backdoor/evaluation.py) evaluates ASR separately from clean loss and accuracy.
Bind its loader and trigger to pass it as `backdoor_evaluation=...` to `run_federated`:

```python
from functools import partial

from meta_sg.attacks.backdoor import evaluate_asr
from meta_sg.datasets.mnist import dba_trigger

backdoor_evaluation = partial(
    evaluate_asr, loader=test_loader, trigger=dba_trigger,
    target_label=2, source_label=None, device=device,
)
```

ASR is the fraction of eligible triggered examples predicted as the target class.
Already-target examples are excluded. For source-specific BFL, set the same `source_label` used by the attack.
Both DBA and BFL are evaluated using the complete trigger. Zero eligible examples make ASR undefined and raise an error.
ASR test labels remain in evaluation and are not passed to attack construction or defense aggregation.

Round records include `attack` diagnostics and optional `backdoor_evaluation` results, including round-zero ASR when configured.
