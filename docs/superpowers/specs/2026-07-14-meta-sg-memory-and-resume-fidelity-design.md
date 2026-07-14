# Meta-SG Memory Stability and Resume Fidelity Design

## Context

The paper-scale global-model-poisoning run completed 10 of 100 outer
iterations before a controlled stop. Lightweight resource telemetry showed a
nearly linear process-RSS increase from about 8 GiB to 18.5 GiB, while GPU
memory stayed near 1 GiB. Source inspection shows that each attack task creates
and destroys multiple `fl_sandbox` coordinators, datasets, data loaders,
models, and short-lived client thread pools. Those objects are no longer part
of the returned task result, but Python cycles and the glibc allocator can keep
their released anonymous pages resident.

The same source inspection found that the current checkpoint format saves TD3
agents but not attacker replay buffers or process RNG states. A model-only
restart is therefore auditable only as a discontinuous training attempt; it is
not equivalent to an uninterrupted 100-iteration run.

## Goals

1. Keep RSS bounded during the H=200, K=10, 30-client experiment without
   changing model updates, random-number consumption, or client parallelism.
2. Make future checkpoints sufficient for continuous training recovery by
   saving all persistent learning and RNG state.
3. Preserve resource, metric, log, configuration, and provenance history when
   a run is resumed.
4. Keep checkpoint storage bounded to one atomically replaced `latest`
   directory every 10 completed outer iterations.
5. Prove the fix with unit tests, a real-backend smoke comparison, and an RSS
   slope gate before committing to another multi-day run.

## Non-goals

- Do not tune the Meta-SG objective, reward, attack mix, or defense policy.
- Do not reduce `parallel_clients` unless task-boundary maintenance fails the
  RSS slope gate.
- Do not treat checkpoint contents as evaluation metrics. Checkpoints are read
  only by save/load verification and the actual recovery path.
- Do not present the stopped 10-iteration diagnostic attempt as part of the
  final paper result.

## Considered approaches

### A. Task-boundary garbage collection and allocator trimming (selected)

After an attack task returns and its coordinator graph is unreachable, run
`gc.collect()` and, on Linux/glibc, best-effort `malloc_trim(0)`. This directly
targets the observed private anonymous RSS while retaining two-client
parallelism and unchanged numerical training behavior. The maintenance call is
opt-in and emits its own timing and RSS telemetry.

### B. `MALLOC_ARENA_MAX=2` only

This is a smaller launcher-only change, but it limits future arenas rather than
explicitly returning already freed pages. It is retained as a fallback if
approach A does not pass the RSS slope gate.

### C. `parallel_clients=1`

This avoids short-lived parallel-client allocations but materially reduces
throughput and changes the execution configuration. It is the final fallback,
not the first fix.

## Design

### Memory-maintenance component

Add a small platform-aware module with one public operation:

```text
perform_memory_maintenance() -> MemoryMaintenanceResult
```

The result contains:

- elapsed seconds;
- objects collected by Python GC;
- RSS before and after, when `/proc/self/status` is available;
- released RSS in KiB;
- whether `malloc_trim` is supported and whether it returned success;
- a bounded warning string for a best-effort platform failure.

The operation must not raise for an unavailable libc symbol or unsupported
platform. Unexpected programming errors remain visible in tests, while runtime
platform limitations become telemetry rather than training failures.

`run_meta_sg_pretraining.py` gains an explicit
`--memory-maintenance {off,task}` option. The general default is `off` for
backward compatibility. The H=200 launcher sets it to `task` by default.

`MetaSGTrainer` calls maintenance after each `AttackTaskRunner.run` has
returned and after task metrics have been copied. Cleanup time is recorded
separately from the existing task compute time and remains part of total outer
iteration wall time. Each task record gets a `memory_maintenance` object, and
the outer record gets totals for maintenance time and released RSS. Existing
iterations without these additive fields remain valid.

### Complete checkpoint state

Continue using the staged-directory plus atomic rename protocol. Extend the
staged checkpoint with:

- defender and attacker TD3 algorithm/optimizer/counter state;
- one HDF5 replay-buffer file per persistent attacker buffer, using Tianshou's
  supported `save_hdf5`/`load_hdf5` API;
- NumPy global RNG state;
- Python `random` state;
- PyTorch CPU RNG state;
- all CUDA RNG states when CUDA is active;
- checkpoint schema version and invariant dimensions/capacities.

Load validates schema and invariant dimensions before mutating the trainer.
Agent, replay-buffer, and RNG state are restored only after all required files
have been validated. Legacy model-only checkpoints require an explicit
`--allow-model-only-resume` flag and emit a provenance discontinuity; they are
never silently described as continuous recovery.

Because the stopped iteration-10 checkpoint predates this format, the final
paper run restarts from iteration 0 with master seed 42 after the fix passes its
gates. The stopped run remains preserved as a diagnostic artifact.

### Resume-safe artifact handling

On a fresh run, the launcher creates headers and immutable initial provenance.
On a resume attempt it:

- appends to the existing resource CSV instead of truncating it;
- appends logs and metrics as today;
- retains initial provenance and appends an attempt record containing time,
  start iteration, command/config digest, reason, and exit status;
- retains status history;
- writes an attempt-specific config file rather than replacing the original
  config without history.

The final evaluator runs only after the trainer reaches iteration 100.

### Failure handling

- Memory maintenance failures are recorded and training continues.
- Any checkpoint validation or serialization failure aborts replacement and
  leaves the previous `latest` intact.
- A non-finite required metric, process death, CUDA OOM, or sustained swap
  growth triggers the existing controlled-stop path.
- If the RSS slope gate fails, test `MALLOC_ARENA_MAX=2` as one isolated
  variable. Only if that also fails should client parallelism be reduced.

## Verification

### Unit and integration tests

1. Linux trim success, unsupported-platform degradation, and libc-symbol
   failure are all covered with mocks.
2. Task maintenance occurs exactly once per completed task and produces
   standards-compliant JSON telemetry.
3. Replay-buffer round-trip preserves length, contents, insertion position,
   and sampling RNG behavior.
4. Trainer checkpoint round-trip preserves agents, replay buffers, NumPy,
   Python, CPU Torch, and CUDA RNG state where available.
5. A resumed deterministic stub run matches an uninterrupted run from the
   checkpoint boundary onward.
6. Latest-only replacement still leaves no history, staging, or backup
   directory after success or rollback.
7. Resume launcher tests prove resource/provenance/config histories append
   rather than truncate.

### Real-backend gates

Run matched small `fl_sandbox` baseline and maintenance-enabled smokes. Then
start the formal seed-42 run and inspect only lightweight training/resource
metrics for the first three outer iterations.

Proceed beyond iteration 3 only if:

- strict JSON metrics are complete and finite;
- task durations remain within normal variance;
- no swap growth occurs;
- post-maintenance RSS does not have a sustained slope above 0.2 GiB per outer
  iteration;
- clean/IPM/LMP/RL final evaluation remains deferred until iteration 100.

## Success criteria

The optimization is complete when a new seed-42 H=200 run reaches 100 outer
iterations with bounded latest-only checkpoints, bounded RSS, append-only
observability artifacts, complete continuous-recovery state, and audited final
clean/IPM/LMP/RL evaluation outputs.
