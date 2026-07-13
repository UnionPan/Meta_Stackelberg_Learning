# Global Model-Poisoning H200 Observable Experiment Design

## Objective

Run one reproducible Meta-Learning experiment over the IPM, LMP, and adaptive-RL
global model-poisoning tasks, retain a bounded recovery checkpoint, and produce a
complete single-seed final evaluation against clean, IPM, LMP, and RL scenarios.
The artifacts must support post-run diagnosis without retaining model-sized data
for every FL round.

This is a controlled single-master-seed experiment. It is suitable for mechanism
analysis and exact reproduction, but it does not estimate cross-seed variance or
statistical significance.

## Existing Entry Points

- `meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh` is the H200 job
  launcher.
- `meta_sg/scripts/run_meta_sg_pretraining.py` creates a timestamped run and
  configures `MetaSGTrainer`.
- `meta_sg/learning/meta_sg_trainer.py` owns iteration metrics and checkpointing.
- `meta_sg/scripts/evaluate_meta_sg_direct.py` evaluates a trained defender on
  clean and model-poisoning scenarios.

The current launcher checkpoints every iteration. The trainer writes both an
`iter_NNNN` directory and `latest`, so storage grows throughout the run. Training
already produces TensorBoard and a JSONL file, but the JSONL omits several
available loss, buffer, task, timing, and per-attack diagnostics. The launcher
does not run final evaluation.

## Experiment Protocol

Use one recorded master seed, defaulting to `42`. Training randomness uses the
master seed. Final evaluation uses a deterministic evaluation stream derived
from the same master seed so it remains reproducible without replaying the exact
training stream. The resolved training and evaluation seeds must both appear in
the provenance and evaluation artifacts.

Meta-training uses the existing paper-scale defaults:

- Dataset: MNIST
- Domain: IPM, LMP, adaptive RL
- Outer iterations: `T=100`
- Tasks per iteration: `K=10`, stratified
- FL horizon: `H=200`
- Defender inner updates: `l=10`
- Attacker best-response updates: `N_A=10`
- Clients/attackers: 30/6
- Client subsampling: 0.2
- Defender action: norm bound, trimmed-mean beta, and NeuroClip epsilon

Final evaluation loads the immutable `final` checkpoint and runs clean, IPM,
LMP, and RL at H=200 with the training data/client/defense configuration. It uses
one evaluation seed stream and native attack behavior. Evaluation is in-domain;
the report must not describe it as held-out attack-family generalization.

## Checkpoint Contract

Checkpointing occurs after iterations 10, 20, ..., 100. Intermediate checkpoint
storage contains exactly one logical checkpoint at `checkpoints/latest`; every
save replaces the previous contents. No `iter_NNNN` directories are created in
latest-only mode.

The latest checkpoint contains the defender and all attacker agents needed for
resume. A small metadata file records the completed global iteration, master
seed, and save time. Replacement must be staged so an interrupted write does not
silently present a partially-written checkpoint as current. The immutable
`final` checkpoint is written after successful training and is not removed when
`latest` is refreshed.

Latest-only behavior is opt-in at the shared trainer/CLI level so existing jobs
that rely on historical `iter_NNNN` checkpoints keep their current behavior.

## Training Observability

`metrics.jsonl` contains one record per completed outer iteration. Each record
includes:

- Global/local iteration and sampled task names.
- Defender reward mean, standard deviation, minimum, and maximum; attacker
  reward mean.
- Mean clean accuracy, backdoor/attack metric when available, and per-attack
  reward summaries.
- Reptile full, actor, and critic update norms, plus inner-adaptation norms.
- Defender TD3 critic loss, actor loss, and Q mean when updates occur.
- Adaptive-attacker TD3 loss/Q summaries when updates occur.
- Defender and attacker decoded actions and raw-action standard deviations.
- Local and attacker replay-buffer sizes, trajectory/transition counts, and
  support update counts.
- Query base/adapted reward, clean/attack metrics, gains, and acceptance
  diagnostics already exposed by the task runner.
- Per-task structured records for IPM, LMP, and RL, retaining available task
  metrics rather than only a cross-task mean.
- Per-task and outer-iteration elapsed time.

TensorBoard mirrors scalar aggregates and per-attack series suitable for live
monitoring. JSON uses explicit `null` for unavailable/non-finite observations
rather than non-standard `NaN` tokens. Logs remain line-buffered so monitoring
can detect stalled work.

## Evaluation Observability

The direct evaluator retains its scenario-level summary and adds a compact
per-round scalar trace for each scenario. Each trace records, when observable:

- Round number.
- Clean accuracy and backdoor/attack metric.
- Defender and attacker reward.
- Decoded defender controls: alpha, beta, NeuroClip epsilon, and server learning
  rate when applicable.
- Server-learning-rate penalty and other scalar environment diagnostics selected
  from a fixed allowlist.

Raw client updates, model tensors, datasets, and arbitrary info objects are not
serialized. Four scenarios at 200 rounds therefore remain small enough for
routine review.

`evaluation/summary.json` aggregates final, mean, minimum, maximum, and worst
round values by scenario and records the exact checkpoint and evaluation seed.
The summary explicitly labels the result as single-seed and supplies no
confidence interval.

## Provenance and Run State

Before training, the launcher writes a provenance record containing:

- Git commit, branch, and dirty status.
- Full resolved command/configuration and seed mapping.
- Python, PyTorch, CUDA, cuDNN, and GPU identity/version information.
- Host, process ID, start time, and relevant device environment.

The timestamped run directory contains separate training and evaluation logs.
`status.json` is updated at stage boundaries (`initializing`, `training`,
`evaluating`, `completed`, or `failed`) with timestamps, exit status, and the
last completed iteration when known. A shell trap preserves failure state and
stops the resource monitor.

## Resource and Progress Monitoring

While the experiment runs, a lightweight monitor periodically appends timestamp,
GPU utilization, memory use, temperature, power, and process memory/CPU to a CSV
file when those observations are available. Absence of `nvidia-smi` must not
abort a CPU/sandbox smoke run.

Operational monitoring checks:

- The training process remains alive.
- Logs and metrics continue to advance within the expected iteration duration.
- GPU memory, temperature, and utilization are plausible.
- New metrics contain no unexpected non-finite values in required fields.
- Checkpoint refreshes occur at the configured 10-iteration cadence.

On a clear failure, invalid required metric, repeated CUDA out-of-memory error,
or process death, stop the active job safely, preserve logs/status/latest, find
and test the root cause, and resume from `checkpoints/latest` with the correct
global iteration. Slow but advancing H200 work is not itself a failure.

## Error Handling

The launcher uses strict shell error handling. Training failure prevents final
evaluation and marks the run failed. Evaluation failure preserves the successful
final checkpoint and training artifacts while marking the evaluation stage
failed. Resume must validate checkpoint files and iteration metadata before
starting; a missing or inconsistent checkpoint fails fast with a clear message.

## Verification

Automated tests cover:

- Latest-only checkpoint mode saves only `latest`, refreshes it at the requested
  interval, and preserves historical mode by default.
- Checkpoint metadata reports the completed global iteration.
- Training JSONL contains aggregate, per-task, loss, buffer, timing, and valid
  JSON nullability fields.
- Direct evaluation emits a 200-round-compatible scalar trace and correct
  scenario summary aggregation in a short stub/synthetic test.
- The H200 launcher defaults to checkpoint interval 10, latest-only retention,
  one master seed, final model-poisoning evaluation, provenance/status/logging,
  and resource monitoring.

Before the full job, run focused unit tests and a tiny CPU/stub smoke experiment
that exercises training, latest replacement, final evaluation wiring, and
artifact layout. The full H200 run starts only after these checks pass.

