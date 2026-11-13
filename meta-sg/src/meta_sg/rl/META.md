# First-order TD3 Meta-SG

This module migrates the update sequence from the original
`scripts/run_paper_stackelberg_td3.py` and `src/meta_sg/rl/reptile_td3.py`.
It uses the cleaned project's datasets, classifier training, root features and
configurable defender observation/action/reward. It does not reproduce the
paper's generated-data world model or claim an exact follower equilibrium.

## Defender update

For each outer iteration, sample `tasks_per_iteration` distinct tasks from
`meta.tasks`. Each task starts with the same shared defender policy weights.

1. Create a task-local defender with fresh replay and optimizer state. Collect
   `inner_steps` episodes against the deterministic task follower. After each
   episode, append defender transitions and execute one TD3 gradient step.
2. Hold that defender fixed. Collect `response_steps` episodes with follower
   exploration and deterministic defender actions. Append only follower
   transitions to its persistent replay and execute one follower TD3 step per
   episode. Each task has an independent follower, replay and optimizer.
3. Create another defender from the adapted weights with fresh replay and
   optimizers. Collect one episode against the updated deterministic follower.
   Fit its critics for `post_critic_steps`, keeping actor and target weights
   fixed. Compute the actor loss gradient `grad(-Q1(s, actor(s)).mean())` on a
   replay minibatch, evaluated at the adapted actor parameters.
4. Update the shared actor: `initial_actor - outer_learning_rate * mean(grad)`.
   Average the post-response critic weights and synchronize both target
   networks. Optimizer and replay states are not averaged.

This is a first-order update: derivatives through task adaptation and follower
response are omitted. It is not Reptile interpolation of adapted actors.
Fresh FL episodes start from the same initial classifier and fixed partitions;
episode seeds give independent participant, augmentation and batch streams.
TD3's `policy_delay` still applies: one gradient step need not update the actor.

`rl.agent` controls defender TD3; `meta.follower_agent` overrides those settings
for followers. Meta collection always uses actor plus exploration noise, so
`learning_starts` is forced to zero for both roles. Explicit block updates use
`inner_steps`, `response_steps` and `post_critic_steps`; `gradient_steps` is only
used by the ordinary single-policy entry. Replay sampling allows replacement
when an episode has fewer transitions than the TD3 minibatch size.

Defender observations include the existing Histogram40 with frozen training
calibration. Its action dimensions follow `rl.action.controls` and `fixed`.
The reward receives only root feedback and observable retained risk.

## Follower settings

Each `meta.tasks` entry contains a unique `name`, `kind`, global
`malicious_clients`, optional `start_round` (default 3), and `parameters`.
Two task kinds are supported:

| Kind | Submitted malicious update | Reward |
| --- | --- | --- |
| `model_poisoning` | Negative scaled SGD delta trained on pooled attacker-owned data | Negative owned clean accuracy |
| `backdoor` | Shared poisoned SGD delta, without model replacement scaling | Owned triggered target accuracy minus owned clean accuracy |

Both use the previous round's Compact8, initially zeros, and three actions in
`[-1, 1]`: scale, SGD learning rate and SGD step count. The default decoder maps
them to `[0, 10]`, `[0, 0.1]` and integer `[1, 11]`. Backdoor ignores the scale
action, matching the original decoder. Bounds are configurable through
`parameters.decoder.maximum_scale`, `maximum_learning_rate`, `maximum_steps`.
Backdoor parameters are `source_label` (default 0; null allows all non-target
labels), `target_label` (default 9) and `poison_fraction` (default 1).

Follower training pools only the named clients' training partitions, excluding
trusted root. Feedback uses deterministic evaluation preprocessing on at most
`feedback_samples` owned clean samples and eligible triggered samples, sampled
per episode. These labels, triggers, identities and rewards do not enter
defender observations or rewards. Task names are orchestration/log metadata.
Test data may be loaded and fingerprinted by the shared dataset setup, but test
evaluation is disabled during every meta episode.

The migrated follower records one transition per physical FL round, including
inactive rounds. Controls execute only after attack onset and when a malicious
client participates. This preserves the physical discount clock; the old
trajectory script compressed follower replay to active rounds. Together with
the cleaned feature/reward interface and new episode seeds, this means old
experiment curves cannot be assumed reproducible from these configurations.
These are learned RL/BRL followers, distinct from the frozen Li2022 attacker.

## Entry and persistence

Install the optional RL dependency, then inspect the example without loading
datasets or calibration files:

```bash
uv sync --group dev --extra rl
uv run meta-sg-meta --config configs/mnist_meta.json --dry-run
```

The example uses a previously fitted Histogram40 transform; see
[calibration instructions](DISTRIBUTIONS.md). It is a wiring example, not a
validated historical experiment protocol. Actual training belongs on the
development machine after a small smoke test and resource checks:

```bash
uv run meta-sg-meta --config configs/mnist_meta.json
uv run meta-sg-meta --resume outputs/mnist_meta/latest.pt
```

The separate entry selects meta-training; ordinary `meta-sg` retains its
single-policy behavior. Use `attack.name=none` and declare attacks in
`meta.tasks`. `meta.horizon` controls FL rounds per episode; `training.rounds`
does not control meta episodes. Other classifier training and partition
settings remain shared. Cross-task training does not start from a dry run.

| Artifact | Contents |
| --- | --- |
| `config.json`, `protocol.json` | Resolved settings, feedback scope, algorithm choices and versions |
| `rounds.jsonl` | Episode/task/phase, client metrics, controls, root feedback and attacker diagnostics |
| `metrics.jsonl` | Phase reward means, actual attacker-active rounds, TD3 update counts/losses, outer gradient norms and per-iteration actor changes/cost counts |
| `status.json` | Status, completed iteration, cumulative episodes and physical FL rounds |
| `latest.pt` | Full defender/follower TD3 states, replay, optimizers, targets, RNGs, partitions, initial classifier, frozen observation and log offsets |
| `latest_fl.pt` | Latest classifier diagnostic snapshot, episode and physical round; not a resume checkpoint |
| `policy.zip`, `policy.json`, `observation.json` | Shared defender exported for existing frozen-policy loading; observation JSON only for frozen distribution observations |

`meta.checkpoint_interval` counts completed outer iterations, default 1;
initial and final states are also saved. `output.checkpoint_interval` separately
controls classifier diagnostic snapshots. Resume restarts from the last saved
outer boundary, discarding and recomputing subsequent partial work and log
suffixes. There is no mid-episode/mid-outer resume. Embedded Histogram40 state
allows resume and frozen inference after the calibration file is removed.

`meta.iterations` is the total outer budget and may be extended at resume.
Other algorithm settings must match; output checkpoint intervals and data root
location may change, but dataset fingerprints must match. Offline cost is
recorded as `iterations * tasks_per_iteration * (inner_steps + response_steps
+ 1) * horizon` physical FL rounds. This entry does not implement deployment
adaptation or its comparison/evaluation protocol.
