# Learnable defense and RL Defender

[defender.py](defender.py) composes client features, replaceable observation/action/reward functions, and a TD3 agent.
All attacks use the same interface. One run has one configured attack and trains its own policy; no cross-attack meta-training is implemented here.

## Configuration

Select `defense.name="rl"`. The [complete example](../../../configs/mnist_rl.json) configures an MNIST/IPM specialist.
Install the optional dependency with `uv sync --group dev --extra rl` before training or loading a policy.

| `rl.mode` | Behavior |
| --- | --- |
| `train` | Collect actual FL transitions and update a TD3 policy |
| `frozen` | Load `rl.checkpoint` (`policy.zip` plus its `policy.json`) and use deterministic actions; no replay or optimization |
| `fixed` | Execute `rl.fixed_action` in the selected control order; default `[1, 0, 1]` for the default mixed-norm component |

Policy optimization runs on the CPU, independently of `training.device`, which selects the classifier device.
`rl.agent` configures learning rate, replay capacity, warmup, batch size, discount, target averaging, policy delay, exploration/target noise, hidden widths, and gradient steps per transition.
`learning_starts` counts warmup actions; updates begin once both that sample budget and `batch_size` are available.
The scalar TD3 learning rate is independent of the classifier's round-wise SGD schedule.

## Feature and action path

[features.py](features.py) probes candidate client updates on the labeled trusted root set using deterministic evaluation transforms and the same global buffers:

| Feature | Value |
| --- | --- |
| C | `1 - cosine(client_update, root_update)` |
| N | Absolute log client/root update norm ratio; zero if either norm is zero |
| B | Maximum absolute change in mean class probabilities |
| Z | Candidate root loss minus global root loss |

Within-round median/MAD deviations, configured floors, and a threshold produce client risks.
`defense.parameters.features` accepts `scale_floors`, `risk_threshold`, and `score_limit`.
The [action components](actions.py) map normalized controls in `[-1,1]` to weights and norm scales. The shared [adaptive mechanism](../defenses/adaptive.py) applies the resulting parameter delta.
Weights normalize the aggregate; norm scaling acts on client deltas before that average. BatchNorm buffers retain the common unweighted synchronization rule.

This migrates the raw features and configurable norm/CNB action mechanisms. The current risk scorer uses within-round robust deviations; historical fitted clean-tail calibration, history/coherence scoring, and NeuroClip are not implemented here. Archived experiment trajectories and policies are not claimed to be exact reproductions.

## Replaceable actions

`rl.action` chooses the action mechanism independently of observations and rewards.

| Name | Full control order | Parameters |
| --- | --- | --- |
| `mixed_norm` (default) | `risk`, `mix`, `strength` | `norm_cap=1`, `maximum_amplification=2` |
| `median_clip` | `risk`, `strength` | `norm_cap=1` |
| `trusted_norm` | `risk`, `strength` | `maximum_amplification=2` |
| `cnb` | `cosine_threshold`, `norm_threshold`, `bias_threshold`, `norm_bound_multiplier` | `threshold_min=.5`, `threshold_max=6`, `norm_bound_min=.5`, `norm_bound_max=5`, `gate_temperature=2`, `scale_floors=[.05,.1,.0001]` |

Mixed norm preserves the existing risk/mix/strength formula. The two-control variants select only the median or root norm reference.
CNB uses lower-median/MAD cosine, two-sided log-norm, and upper bias anomalies; its four controls decode to soft thresholds and a median-norm bound as in the existing CNB adapter. It uses the shared raw client features and is independent of the observation summary.

`controls` selects which named controls the policy learns and in what order; omitted controls must have explicit normalized values in `fixed`. Omitting both selects all controls in the component's full order.
For example, to learn only risk and norm strength while keeping the norm-reference mix at its midpoint, set `rl.action` to:

```json
{
  "name": "mixed_norm",
  "controls": ["risk", "strength"],
  "fixed": {"mix": 0},
  "parameters": {"norm_cap": 1, "maximum_amplification": 2}
}
```

This produces a two-dimensional policy. Use `mix=-1` for the median reference or `mix=1` for the root reference. TD3 output, warmup/exploration noise, replay arrays, frozen loading, and checkpoint recovery use the selected dimension automatically.
In `fixed` mode, `rl.fixed_action` must follow `controls` order and length. Training/frozen modes require at least one selected control.
`--dry-run` reports the selected names and dimension; per-round diagnostics also record `action_names`.

To add a mechanism, register `ACTIONS['name'] = (('control_a', 'control_b'), function)` in [actions.py](actions.py).
The stateless function receives `(updates, context, full_action, **parameters)` and returns `(parameter_delta, diagnostics)`, with `diagnostics['retained_risk']` for the reward interface. It receives no attack identities or test labels.
The component boundary expands fixed controls before calling it, so it always receives its full named vector.
Action settings, control order, fixed values, and parameters are checked when loading a frozen policy or resuming a run.
Previous default three-action exports/checkpoints remain supported; the earlier `defense.parameters.action` dictionary is normalized into `rl.action.parameters` when loading configurations.

## Replaceable observations

Functions in [observations.py](observations.py) receive a context dictionary with `features`, `risks`, `norms`, `root_norm`, `before`, and `score_limit` and return a one-dimensional `float32` array.

| Name | Parameters | Output |
| --- | --- | --- |
| `compact8` | None | Raw C/N/B/Z medians and 90th percentiles, matching the existing compact summary formula |
| `quantiles` | `quantiles`, `columns` | Selected raw-feature quantiles |
| `statistics` | `columns` | Mean, standard deviation, minimum, maximum |
| `histogram` | `bins`, per-column `ranges` | Fixed-bin fractions, 40 values by default |
| `histogram40` | `transform_path`, `signed_log_norm` | Original ten-bin summary with training-fitted boundaries and normalization |

`histogram` is the simplified stateless summary. The original `histogram40` is described in [Histogram40](DISTRIBUTIONS.md), including training-only fitting, legacy file loading, and persistence.
Round diagnostics retain raw `client_features` and `client_norms`, so a training calibration run's `metrics.jsonl` can directly supply the fitter without reconstructing distributions from compact summaries.
For example, set `rl.observation` to:

```json
{"name": "quantiles", "parameters": {"quantiles": [0.25, 0.5, 0.75], "columns": [0, 1, 2, 3]}}
```

## Replaceable rewards

Functions in [rewards.py](rewards.py) receive only `before` and `after` trusted-root loss/accuracy plus `retained_risk`, and return a scalar.

| Name | Formula and parameters |
| --- | --- |
| `root_loss_delta` | `scale * (before_loss - after_loss) / (1 + before_loss)` |
| `utility_delta` | Scaled weighted accuracy gain + normalized loss gain - retained risk; `accuracy_weight`, `loss_weight`, `risk_weight`, `scale` |
| `absolute_cost` | Negative scaled weighted after-loss and retained risk; `loss_weight`, `risk_weight`, `scale` |

Retained risk is the effective-weighted mean of normalized client risks, not a known malicious-client fraction.
For example, set `rl.reward` to:

```json
{"name": "utility_delta", "parameters": {"accuracy_weight": 1, "loss_weight": 1, "risk_weight": 0.5, "scale": 10}}
```

Root labels are trusted training feedback. Attack identity, malicious identities, triggers, ASR, and test labels are not passed into either component.
Test accuracy and ASR remain separate offline evaluation fields.

To add an observation or reward component, implement a stateless function with this contract and add it to `OBSERVATIONS` or `REWARDS`. Python callers can also inject `observation=` and `reward=` into `RLDefender` directly. No agent or federated-loop changes are needed.
Changing components starts a new experiment; resume requires unchanged observation/action/reward settings.

## Meta-SG

[First-order TD3 Meta-SG](META.md) adds task-local defender adaptation, persistent
task-specific follower responses and an outer actor gradient update through the
separate `meta-sg-meta` entry. It reuses these components and exports a compatible
frozen defender policy. Deployment adaptation is not part of that entry.

## Transitions, export, and recovery

Each completed FL round supplies one reward from the updated global model on the root set.
The following round's actual client features complete its transition; the final transition uses a terminal flag and zero next observation.
This is online fixed-attack training on the shared FL loop. Historical offline response banks and lookahead reward training are not migrated by this entry point.

At each experiment checkpoint, `latest.pt` includes actor/critic/targets, Adam state, replay contents and cursor, private optimization/exploration RNGs, and any pending transition.
Training-frozen distribution components also include fitted observation state in the checkpoint and policy export, with a separate `observation.json` copy.
`policy.zip` and `policy.json` export a separate frozen-inference artifact, including action component settings and selected names. Loading checks observation and action configuration compatibility.
Extending a completed RL run preserves its earlier episode boundary; it is not identical to a run configured with the longer horizon from the start.
See [experiment persistence](../experiments/README.md#saving-and-resuming).
