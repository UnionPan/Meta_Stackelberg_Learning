# Global Model-Poisoning H200 Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and run one reproducible H200 Meta-Learning experiment whose only rolling checkpoint is atomically refreshed every ten outer iterations and whose training, resource, provenance, and final clean/IPM/LMP/RL evaluation artifacts are sufficient for post-run review.

**Architecture:** Keep shared behavior backward-compatible by adding opt-in latest-only retention and exact run-directory flags to the Python trainer entry point. Enrich the trainer's existing JSONL/TensorBoard emission from already-available `TaskResult` values, add compact scalar round traces and aggregation to the existing direct evaluator, and make the H200 shell launcher the stage orchestrator for provenance, status, resource monitoring, logs, training, and final evaluation.

**Tech Stack:** Bash, Python 3, PyTorch/TD3, JSON/JSONL, TensorBoard, pytest, `nvidia-smi` when available.

---

## File Structure

- Modify `meta_sg/learning/meta_sg_trainer.py`: latest-only staged checkpoint replacement, checkpoint metadata, detailed training records, JSON-safe serialization, timing, and TensorBoard additions.
- Modify `meta_sg/scripts/run_meta_sg_pretraining.py`: CLI plumbing for latest-only retention and an exact run name.
- Modify `meta_sg/scripts/evaluate_meta_sg_direct.py`: honor the requested evaluation horizon, retain compact per-round metrics, write JSON-safe output, and optionally write an aggregate summary.
- Create `meta_sg/scripts/experiment_artifacts.py`: provenance and stage-status JSON writer used by the launcher.
- Modify `meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh`: one-master-seed orchestration, latest-only ten-iteration checkpoints, logs, monitor, final evaluation, and failure traps.
- Modify `meta_sg/tests/test_meta_sg_pretraining.py`: trainer, metrics, evaluator, artifact-helper, and launcher contract tests.

### Task 1: Latest-Only Recoverable Checkpointing

**Files:**
- Modify: `meta_sg/learning/meta_sg_trainer.py`
- Modify: `meta_sg/scripts/run_meta_sg_pretraining.py`
- Test: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Write the failing latest-only checkpoint test**

Add a stub-backend test that runs two outer iterations with interval one and the new flag:

```python
def test_pretraining_latest_only_checkpoint_replaces_history_and_records_iteration(tmp_path):
    from meta_sg.scripts.run_meta_sg_pretraining import main

    main([
        "--backend", "stub", "--output-dir", str(tmp_path), "--run-name", "run",
        "--T", "2", "--K", "1", "--H", "1", "--l", "1", "--N-A", "1",
        "--post-br-defender-updates", "0", "--hidden-dim", "8",
        "--batch-size", "2", "--buffer-capacity", "32", "--num-clients", "6",
        "--num-attackers", "1", "--subsample-rate", "1.0", "--seed", "42",
        "--device", "cpu", "--checkpoint-interval", "1", "--latest-checkpoint-only",
    ])

    checkpoint_root = tmp_path / "run" / "checkpoints"
    assert sorted(path.name for path in checkpoint_root.iterdir()) == ["latest"]
    metadata = json.loads((checkpoint_root / "latest" / "checkpoint.json").read_text())
    assert metadata["completed_iteration"] == 2
    assert metadata["master_seed"] == 42
    assert (checkpoint_root / "latest" / "defender_meta.pt").exists()
```

- [ ] **Step 2: Run the test and verify the missing CLI flag fails**

Run:

```bash
.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py::test_pretraining_latest_only_checkpoint_replaces_history_and_records_iteration -q
```

Expected: `SystemExit: 2` because `--latest-checkpoint-only` and `--run-name` do not exist.

- [ ] **Step 3: Add opt-in CLI and trainer fields**

Add parser arguments:

```python
parser.add_argument("--run-name", default="", help="Exact child directory under --output-dir; timestamped when empty.")
parser.add_argument(
    "--latest-checkpoint-only",
    action="store_true",
    help="At each checkpoint interval, stage and replace checkpoints/latest without iter_NNNN history.",
)
```

Resolve output with:

```python
run_name = str(args.run_name).strip() or time.strftime("%Y%m%d-%H%M%S")
output_dir = Path(args.output_dir) / run_name
```

Pass `checkpoint_latest_only=args.latest_checkpoint_only` and
`checkpoint_master_seed=args.seed` to `MetaSGTrainer`.

- [ ] **Step 4: Implement staged latest replacement and metadata**

Extend the trainer constructor with backward-compatible defaults. Split save into
`_write_checkpoint_directory` and `_replace_latest_checkpoint`. Stage a complete
directory, rotate the old `latest` to a hidden backup, promote staging with
`os.replace`, restore the backup on caught failure, and remove the backup only
after promotion. Write this metadata with the agents:

```python
{
    "completed_iteration": int(completed_iteration),
    "master_seed": int(self.checkpoint_master_seed),
    "saved_at": datetime.now(timezone.utc).isoformat(),
}
```

At checkpoint cadence:

```python
if self.checkpoint_latest_only:
    self.save(latest_path, completed_iteration=t + 1, replace=True)
else:
    self.save(iteration_path, completed_iteration=t + 1)
    self.save(latest_path, completed_iteration=t + 1, replace=True)
```

The ordinary final save remains a normal immutable directory.

- [ ] **Step 5: Verify latest-only and historical modes**

Run the new test and the existing traceable-checkpoint/resume tests. Expected:
all pass, latest-only creates no `iter_*`, and default mode still creates them.

- [ ] **Step 6: Commit the checkpoint slice**

```bash
git add meta_sg/learning/meta_sg_trainer.py meta_sg/scripts/run_meta_sg_pretraining.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: add bounded latest-only Meta-SG checkpoints"
```

### Task 2: Complete Per-Iteration Training Records

**Files:**
- Modify: `meta_sg/learning/meta_sg_trainer.py`
- Test: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Expand the traceability test first**

Require each metrics record to contain finite/nullable JSON-safe aggregate and
per-task fields:

```python
record = records[-1]
assert {"reward_min", "reward_max", "iteration_elapsed_seconds"} <= record.keys()
assert {"defender_losses", "attacker_losses", "buffer_sizes", "task_records"} <= record.keys()
assert len(record["task_records"]) == 1
task = record["task_records"][0]
assert {"attack_type", "elapsed_seconds", "transitions_collected", "diagnostics"} <= task.keys()
json.dumps(record, allow_nan=False)
```

- [ ] **Step 2: Run and observe the expected missing-key failure**

Run the focused existing test. Expected: assertion failure on `reward_min`.

- [ ] **Step 3: Capture task and iteration timing**

In `train`, append each `time.perf_counter() - task_started_at` to a list and
compute the outer elapsed time once. Pass those values into `_log_all` and
`_write_metrics_record`.

- [ ] **Step 4: Serialize all bounded observable task metrics**

Add helpers that turn `TaskResult` into a record containing reward totals,
trajectory/transition counts, defender losses, attacker losses, inner delta,
query fields, diagnostics, and elapsed time. Add aggregate loss, buffer,
inner-delta, reward min/max, and timing objects. Do not serialize parameters,
replay contents, trajectories, or arbitrary environment objects.

- [ ] **Step 5: Enforce standards-compliant JSON**

Add a recursive helper:

```python
def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value
```

Write JSONL with `json.dumps(_json_safe(record), allow_nan=False, sort_keys=True)`.

- [ ] **Step 6: Mirror missing scalar series to TensorBoard**

Add outer/task elapsed time, inner delta, per-attack clean/attack metrics, loss
aggregates, and buffer occupancy without changing existing tag names.

- [ ] **Step 7: Run focused and full Meta-SG tests**

```bash
.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q
```

Expected: pass with no JSON `NaN` values.

- [ ] **Step 8: Commit the observability slice**

```bash
git add meta_sg/learning/meta_sg_trainer.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: record complete Meta-SG iteration diagnostics"
```

### Task 3: Round-Level Final Evaluation and Summary

**Files:**
- Modify: `meta_sg/scripts/evaluate_meta_sg_direct.py`
- Test: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Write failing trace and summary tests**

Use a fake short environment/defender around `_evaluate_scenario_at` and a pure
summary helper. Assert that the requested `horizon`, not `args.H`, controls the
loop, that each row is scalar-only, and that aggregation is correct:

```python
assert len(record["round_metrics"]) == 3
assert record["round_metrics"][-1]["round"] == 3
summary = summarize_evaluation_records([record], checkpoint="final", master_seed=42, evaluation_seed=10042)
assert summary["single_seed"] is True
assert summary["scenarios"][record["scenario"]]["rounds"] == 3
json.dumps(summary, allow_nan=False)
```

- [ ] **Step 2: Run tests and verify the missing trace/helper failure**

Expected: missing `round_metrics` or import failure for the summary helper.

- [ ] **Step 3: Record compact per-round scalars**

During evaluation append round number, clean/backdoor accuracy, both rewards,
decoded defense controls, and allowlisted finite scalar info. Change the loop to
`range(int(horizon))`. Never include model/update tensors, loaders, or decision
objects.

- [ ] **Step 4: Add JSON-safe scenario aggregation**

Implement `summarize_evaluation_records` with exact checkpoint, master/evaluation
seeds, `single_seed: true`, no confidence interval, and scenario final/mean/min/
max/worst-round statistics. Add optional CLI arguments `--summary-json` and
`--master-seed`; write both outputs with `allow_nan=False`.

- [ ] **Step 5: Run evaluator and Meta-SG tests**

Run the new unit tests plus existing scenario/evaluator tests. Expected: pass.

- [ ] **Step 6: Commit the evaluation slice**

```bash
git add meta_sg/scripts/evaluate_meta_sg_direct.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: retain round-level Meta-SG evaluation metrics"
```

### Task 4: Provenance, Status, and H200 Job Orchestration

**Files:**
- Create: `meta_sg/scripts/experiment_artifacts.py`
- Modify: `meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh`
- Test: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Write failing helper and launcher-contract tests**

Test that status writes a history without discarding earlier stages and that the
launcher contains the required defaults and invocations:

```python
assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"' in text
assert "--latest-checkpoint-only" in text
assert "--summary-json" in text
assert "experiment_artifacts.py provenance" in text
assert "experiment_artifacts.py status" in text
assert "resource_metrics.csv" in text
assert "--scenario-set model_poisoning" in text
```

- [ ] **Step 2: Run tests and verify helper/contract failures**

Expected: module import failure and current interval `1` mismatch.

- [ ] **Step 3: Implement the artifact helper**

Provide two subcommands:

```bash
experiment_artifacts.py provenance --output RUN/provenance.json --repo-root ROOT \
  --master-seed 42 --training-seed 42 --evaluation-seed 10042 --device cuda:0
experiment_artifacts.py status --output RUN/status.json --stage training --message started
```

Provenance records git commit/branch/dirty paths, interpreter and package/CUDA/
GPU details, host/PID/time, seeds, device, and relevant environment. Status reads
the prior file, appends a timestamped stage entry, and atomically replaces the
JSON file.

- [ ] **Step 4: Turn the H200 launcher into a strict stage orchestrator**

Use `MASTER_SEED=42`, a deterministic evaluation offset, exact `RUN_ID`, and
`RUN_DIR`. Set interval ten and pass `--latest-checkpoint-only --run-name
"$RUN_ID"`. Start training in the background with output tee'd to
`logs/train.log`, monitor its PID, and wait for its true exit code. On success,
run `evaluate_meta_sg_direct.py` against `final` with `--scenario-set
model_poisoning --H 200`, matched client/data/defense settings, the derived one
evaluation seed for both fixed and RL scenarios, raw JSON output, and summary
JSON output.

- [ ] **Step 5: Add resource monitoring and traps**

Write a header and periodic timestamp/process CPU/RSS plus `nvidia-smi` GPU
utilization, memory, temperature, and power to `resource_metrics.csv`. A cleanup
trap stops the monitor. TERM/INT safely stop the active child. EXIT writes
`failed` unless the stage reached `completed`. Missing `nvidia-smi` yields blank
GPU columns and does not fail the job.

- [ ] **Step 6: Run helper and static launcher tests**

Expected: provenance/status valid JSON, stage history preserved, launcher
contract passes, and `bash -n` reports no syntax errors.

- [ ] **Step 7: Commit the orchestration slice**

```bash
git add meta_sg/scripts/experiment_artifacts.py meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: orchestrate observable H200 poisoning experiment"
```

### Task 5: Verification, Smoke Run, and Full Experiment Launch

**Files:**
- Verify: all files above
- Artifacts: `fl_sandbox/runs/meta_sg_global_h200_t100_k10_c30_a6/<RUN_ID>/`

- [ ] **Step 1: Run focused automated verification**

```bash
bash -n meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run a tiny CPU/stub end-to-end smoke**

Add a launcher-supported `BACKEND=stub`/small environment override if needed,
then run `T=2 K=1 H=2 L=1 N_A=1 CHECKPOINT_INTERVAL=1 DEVICE=cpu` into a temporary
run root. Expected: provenance/status/logs/metrics/latest/final/evaluation and
summary files all exist, latest contains iteration 2, and status is completed.

- [ ] **Step 3: Audit smoke artifacts programmatically**

Load every JSON with strict parsing, require two JSONL iterations, verify no
`iter_*` directories, verify round-trace lengths match the smoke horizon, and
verify provenance seeds/config paths.

- [ ] **Step 4: Start the full H200 run**

```bash
TENSORBOARD_HOST=127.0.0.1 TENSORBOARD_PORT=6008 \
  bash meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
```

Keep the unified execution session active so progress can be polled.

- [ ] **Step 5: Monitor progress on a regular cadence**

At each check, inspect the live process, recent training log, newest JSONL record,
file modification times, resource CSV tail, GPU state, and checkpoint cadence.
Report completed iteration, elapsed iteration time, reward/accuracy/loss health,
GPU utilization/memory/temperature, and estimated trend without claiming a
precise completion time too early.

- [ ] **Step 6: Stop, diagnose, test, and resume real failures**

For process death, CUDA OOM, invalid required metric, corrupted/stale output, or
repeated runtime exception: terminate the child safely, preserve the run, use
the systematic-debugging workflow, add a failing regression test, fix and verify,
then resume from `checkpoints/latest` using its recorded completed iteration.

- [ ] **Step 7: Complete the final artifact audit**

After evaluation, prove every design requirement from the actual run: 100 JSONL
records, only latest rolling checkpoint at iteration 100, immutable final
checkpoint, completed status, provenance/resource logs, four 200-round scenario
traces, strict JSON, and aggregate evaluation summary. Only then mark the goal
complete and report paths and headline results.
