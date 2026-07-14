# Meta-SG Memory and Resume Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the H=200 Meta-SG run's host RSS bounded and make every latest-only checkpoint sufficient for deterministic, auditable continuous recovery.

**Architecture:** Add an opt-in task-boundary memory-maintenance service and feed its bounded telemetry into existing task and iteration JSON records. Extend the atomic checkpoint directory with replay buffers, RNG state, a versioned manifest, and validate-before-mutate loading; then make training and launcher artifacts append by attempt so a resumed process cannot erase history.

**Tech Stack:** Python 3.11, PyTorch, NumPy, Tianshou 2.x HDF5 replay buffers, pytest, Bash, `/proc`, glibc `malloc_trim`, JSON/JSONL/CSV.

---

## File structure

- Create `meta_sg/learning/memory_maintenance.py`: platform-aware GC, RSS reading, and best-effort allocator trimming with JSON-safe results.
- Create `meta_sg/tests/test_memory_maintenance.py`: isolated unit tests for Linux success and graceful platform failures.
- Create `meta_sg/tests/test_meta_sg_checkpoint_resume.py`: replay-buffer, full checkpoint, validation, rollback, and deterministic-resume tests.
- Modify `meta_sg/learning/replay_buffer.py`: stable HDF5 save/load methods plus invariant validation.
- Modify `meta_sg/learning/meta_sg_trainer.py`: task-boundary maintenance, full checkpoint manifests/state, legacy opt-in, and telemetry aggregation.
- Modify `meta_sg/scripts/run_meta_sg_pretraining.py`: CLI options, Python RNG seeding, attempt-specific configuration, and resume validation wiring.
- Modify `meta_sg/scripts/experiment_artifacts.py`: immutable provenance creation and append-only attempt lifecycle records.
- Modify `meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh`: task maintenance by default, resume-safe CSV/provenance/config behavior, and attempt completion records.
- Modify `meta_sg/tests/test_meta_sg_pretraining.py`: trainer CLI, artifact helper, and launcher contract/integration coverage.

### Task 1: Platform-safe memory maintenance

**Files:**
- Create: `meta_sg/learning/memory_maintenance.py`
- Create: `meta_sg/tests/test_memory_maintenance.py`

- [ ] **Step 1: Write failing result and Linux-success tests**

```python
from meta_sg.learning import memory_maintenance as mm


def test_memory_maintenance_collects_and_trims(monkeypatch):
    rss_values = iter([20_000, 12_000])
    monkeypatch.setattr(mm, "_read_rss_kib", lambda: next(rss_values))
    monkeypatch.setattr(mm.gc, "collect", lambda: 17)
    monkeypatch.setattr(mm, "_malloc_trim", lambda: (True, True, None))

    result = mm.perform_memory_maintenance()

    assert result.objects_collected == 17
    assert result.rss_before_kib == 20_000
    assert result.rss_after_kib == 12_000
    assert result.rss_released_kib == 8_000
    assert result.malloc_trim_supported is True
    assert result.malloc_trim_succeeded is True
    assert result.warning is None
    assert result.elapsed_seconds >= 0.0
    assert result.as_dict()["rss_released_kib"] == 8_000
```

- [ ] **Step 2: Write failing degradation tests**

```python
def test_memory_maintenance_degrades_when_trim_is_unsupported(monkeypatch):
    monkeypatch.setattr(mm, "_read_rss_kib", lambda: None)
    monkeypatch.setattr(mm.gc, "collect", lambda: 0)
    monkeypatch.setattr(mm, "_malloc_trim", lambda: (False, False, "unsupported platform"))

    result = mm.perform_memory_maintenance()

    assert result.rss_before_kib is None
    assert result.rss_after_kib is None
    assert result.rss_released_kib is None
    assert result.malloc_trim_supported is False
    assert result.malloc_trim_succeeded is False
    assert result.warning == "unsupported platform"


def test_malloc_trim_symbol_failure_is_bounded(monkeypatch):
    monkeypatch.setattr(mm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mm.ctypes, "CDLL", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")))

    supported, succeeded, warning = mm._malloc_trim()

    assert supported is False
    assert succeeded is False
    assert warning is not None
    assert len(warning) <= 240
```

- [ ] **Step 3: Run tests and verify the module is missing**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_memory_maintenance.py -q`

Expected: collection fails because `meta_sg.learning.memory_maintenance` does not exist.

- [ ] **Step 4: Implement the focused maintenance module**

```python
from __future__ import annotations

import ctypes
import gc
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryMaintenanceResult:
    elapsed_seconds: float
    objects_collected: int
    rss_before_kib: int | None
    rss_after_kib: int | None
    rss_released_kib: int | None
    malloc_trim_supported: bool
    malloc_trim_succeeded: bool
    warning: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _bounded_warning(value: object) -> str:
    return str(value).replace("\n", " ")[:240]


def _read_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _malloc_trim() -> tuple[bool, bool, str | None]:
    if platform.system() != "Linux":
        return False, False, "malloc_trim unsupported on this platform"
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return True, bool(trim(0)), None
    except (AttributeError, OSError, TypeError) as exc:
        return False, False, _bounded_warning(exc)


def perform_memory_maintenance() -> MemoryMaintenanceResult:
    started = time.perf_counter()
    before = _read_rss_kib()
    collected = int(gc.collect())
    supported, succeeded, warning = _malloc_trim()
    after = _read_rss_kib()
    released = None if before is None or after is None else before - after
    return MemoryMaintenanceResult(
        elapsed_seconds=time.perf_counter() - started,
        objects_collected=collected,
        rss_before_kib=before,
        rss_after_kib=after,
        rss_released_kib=released,
        malloc_trim_supported=supported,
        malloc_trim_succeeded=succeeded,
        warning=warning,
    )
```

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_memory_maintenance.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Commit the component**

```bash
git add meta_sg/learning/memory_maintenance.py meta_sg/tests/test_memory_maintenance.py
git commit -m "feat: add task memory maintenance telemetry"
```

### Task 2: Execute maintenance exactly once after each task

**Files:**
- Modify: `meta_sg/learning/meta_sg_trainer.py`
- Modify: `meta_sg/scripts/run_meta_sg_pretraining.py`
- Modify: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Add a failing CLI default test**

```python
def test_pretraining_memory_maintenance_defaults_off():
    from meta_sg.scripts.run_meta_sg_pretraining import parse_args

    assert parse_args([]).memory_maintenance == "off"
    assert parse_args(["--memory-maintenance", "task"]).memory_maintenance == "task"
```

- [ ] **Step 2: Add a failing trainer integration test**

Use the existing stub smoke fixture to run `T=1`, `K=2`, inject a callable returning distinct `MemoryMaintenanceResult` values, then assert:

```python
assert maintenance.call_count == 2
record = json.loads(metrics_path.read_text().splitlines()[0])
assert len(record["task_records"]) == 2
assert all("memory_maintenance" in task for task in record["task_records"])
assert record["memory_maintenance"]["calls"] == 2
assert record["memory_maintenance"]["objects_collected"] == 6
assert record["memory_maintenance"]["rss_released_kib"] == 3072
assert record["memory_maintenance"]["elapsed_seconds"] == pytest.approx(0.03)
json.dumps(record, allow_nan=False)
```

- [ ] **Step 3: Run the focused tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q -k 'memory_maintenance'`

Expected: FAIL because the CLI and trainer injection point do not exist.

- [ ] **Step 4: Add the CLI and trainer dependency**

Add to `parse_args`:

```python
parser.add_argument(
    "--memory-maintenance",
    choices=("off", "task"),
    default="off",
    help="Run Python GC and best-effort allocator trimming after each completed task.",
)
```

Add to `MetaSGTrainer.__init__`:

```python
memory_maintenance: Callable[[], MemoryMaintenanceResult] | None = None,
```

Wire it in `main` only when enabled:

```python
memory_maintenance=(
    perform_memory_maintenance if args.memory_maintenance == "task" else None
),
```

- [ ] **Step 5: Record maintenance after copied task output**

In the task loop, keep `task_elapsed_seconds` as compute-only time and call maintenance after appending the result:

```python
task_elapsed = time.perf_counter() - task_started_at
task_elapsed_seconds.append(task_elapsed)
maintenance = (
    self.memory_maintenance()
    if self.memory_maintenance is not None
    else None
)
task_memory_maintenance.append(maintenance)
```

Pass the new list into `_log_all` and `_write_metrics_record`. Emit per-task `result.as_dict()` and an outer aggregate whose nullable RSS sum ignores unavailable samples but remains `None` when every sample is unavailable.

- [ ] **Step 6: Run focused integration and existing metrics tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_memory_maintenance.py meta_sg/tests/test_meta_sg_pretraining.py -q -k 'memory_maintenance or traceable_metrics'`

Expected: all selected tests pass and the generated JSON rejects NaN.

- [ ] **Step 7: Commit trainer integration**

```bash
git add meta_sg/learning/meta_sg_trainer.py meta_sg/scripts/run_meta_sg_pretraining.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: maintain memory at Meta-SG task boundaries"
```

### Task 3: Persist and validate Tianshou replay buffers

**Files:**
- Modify: `meta_sg/learning/replay_buffer.py`
- Create: `meta_sg/tests/test_meta_sg_checkpoint_resume.py`

- [ ] **Step 1: Write a failing round-trip test**

```python
def test_replay_buffer_hdf5_round_trip_preserves_state_and_sampling(tmp_path):
    source = ReplayBuffer(capacity=7, obs_dim=2, act_dim=1)
    for index in range(10):
        source.add(
            np.array([index, index + 0.5], dtype=np.float32),
            np.array([index / 10], dtype=np.float32),
            float(index),
            np.array([index + 1, index + 1.5], dtype=np.float32),
            index % 3 == 0,
        )
    path = tmp_path / "buffer.hdf5"
    source.save(path)
    restored = ReplayBuffer.load(path, capacity=7, obs_dim=2, act_dim=1)

    assert len(restored) == len(source) == 7
    assert restored.capacity == source.capacity
    np.random.seed(123)
    expected = source.sample(5)
    np.random.seed(123)
    actual = restored.sample(5)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
```

- [ ] **Step 2: Write failing invariant tests**

```python
@pytest.mark.parametrize(
    ("capacity", "obs_dim", "act_dim"),
    [(8, 2, 1), (7, 3, 1), (7, 2, 2)],
)
def test_replay_buffer_load_rejects_invariant_mismatch(tmp_path, capacity, obs_dim, act_dim):
    source = ReplayBuffer(capacity=7, obs_dim=2, act_dim=1)
    source.add(np.zeros(2), np.zeros(1), 0.0, np.ones(2), False)
    path = tmp_path / "buffer.hdf5"
    source.save(path)

    with pytest.raises(ValueError, match="replay buffer invariant"):
        ReplayBuffer.load(path, capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py -q -k replay_buffer`

Expected: FAIL because `ReplayBuffer.save` and `ReplayBuffer.load` do not exist.

- [ ] **Step 4: Implement typed HDF5 persistence**

```python
def save(self, path: str | Path) -> None:
    self._buffer.save_hdf5(str(path))

@classmethod
def load(
    cls,
    path: str | Path,
    *,
    capacity: int,
    obs_dim: int,
    act_dim: int,
) -> "ReplayBuffer":
    loaded = TianshouReplayBuffer.load_hdf5(str(path))
    if int(loaded.maxsize) != int(capacity):
        raise ValueError("replay buffer invariant mismatch: capacity")
    result = cls(capacity=capacity, obs_dim=obs_dim, act_dim=act_dim)
    result._buffer = loaded
    if len(result):
        sample = result._buffer[result._buffer.sample_indices(1)]
        if np.asarray(sample.obs).shape[-1] != obs_dim:
            raise ValueError("replay buffer invariant mismatch: obs_dim")
        if np.asarray(sample.act).shape[-1] != act_dim:
            raise ValueError("replay buffer invariant mismatch: act_dim")
    return result
```

- [ ] **Step 5: Run replay-buffer tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py -q -k replay_buffer`

Expected: all replay-buffer tests pass, including wrapped circular state and identical seeded sampling.

- [ ] **Step 6: Commit replay persistence**

```bash
git add meta_sg/learning/replay_buffer.py meta_sg/tests/test_meta_sg_checkpoint_resume.py
git commit -m "feat: persist Meta-SG attacker replay buffers"
```

### Task 4: Save a versioned complete checkpoint and validate before mutation

**Files:**
- Modify: `meta_sg/learning/meta_sg_trainer.py`
- Modify: `meta_sg/tests/test_meta_sg_checkpoint_resume.py`

- [ ] **Step 1: Write a failing complete-state manifest test**

Create a tiny trainer, add transitions to every attacker buffer, save iteration 3, and assert:

```python
metadata = json.loads((checkpoint / "checkpoint.json").read_text())
assert metadata["schema_version"] == 2
assert metadata["completed_iteration"] == 3
assert metadata["invariants"] == {
    "obs_dim": trainer.obs_dim,
    "defender_act_dim": trainer.act_dim,
    "attacker_act_dim": trainer.attacker_act_dim,
    "buffer_capacity": trainer.td3_cfg.buffer_capacity,
    "attackers": sorted(trainer.attacker_agents),
}
assert (checkpoint / "rng_state.pt").exists()
for name in trainer.attacker_agents:
    assert (checkpoint / f"attacker_buffer_{name}.hdf5").exists()
```

- [ ] **Step 2: Write a failing RNG and buffer restore test**

Capture the expected next values after saving, perturb every generator and buffer, then load and assert exact equality:

```python
expected_python = random.random()
expected_numpy = np.random.random(5)
expected_torch = torch.rand(5)
trainer.load(checkpoint)
assert random.random() == expected_python
np.testing.assert_array_equal(np.random.random(5), expected_numpy)
torch.testing.assert_close(torch.rand(5), expected_torch, rtol=0, atol=0)
assert {name: len(buf) for name, buf in trainer.attacker_buffers.items()} == saved_lengths
```

On CUDA hosts, also assert `torch.cuda.get_rng_state_all()` round-trips; skip only the CUDA assertion when unavailable.

- [ ] **Step 3: Write a failing validate-before-mutate test**

```python
before = {key: value.clone() for key, value in trainer.defender.get_params().items()}
metadata["invariants"]["obs_dim"] += 1
(checkpoint / "checkpoint.json").write_text(json.dumps(metadata))

with pytest.raises(ValueError, match="obs_dim"):
    trainer.load(checkpoint)
for key, value in before.items():
    torch.testing.assert_close(trainer.defender.get_params()[key], value)
```

- [ ] **Step 4: Run tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py -q -k 'checkpoint and not legacy'`

Expected: FAIL because the schema, RNG file, buffers, and validation are absent.

- [ ] **Step 5: Implement schema-2 writes inside the existing staged directory**

Add constants and save the full state before writing the manifest last:

```python
CHECKPOINT_SCHEMA_VERSION = 2

rng_state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
}
torch.save(rng_state, Path(directory, "rng_state.pt"))
for name, buffer in self.attacker_buffers.items():
    buffer.save(Path(directory, f"attacker_buffer_{name}.hdf5"))
metadata = {
    "schema_version": CHECKPOINT_SCHEMA_VERSION,
    "completed_iteration": int(completed_iteration),
    "master_seed": self.checkpoint_master_seed,
    "saved_at": datetime.now(timezone.utc).isoformat(),
    "invariants": {
        "obs_dim": self.obs_dim,
        "defender_act_dim": self.act_dim,
        "attacker_act_dim": self.attacker_act_dim,
        "buffer_capacity": self.td3_cfg.buffer_capacity,
        "attackers": sorted(self.attacker_agents),
    },
}
```

- [ ] **Step 6: Implement validation and all-or-nothing restore ordering**

Create `_validate_checkpoint(directory) -> ValidatedCheckpoint` that parses JSON, checks schema and exact invariants, checks every required file, constructs temporary defender/attacker `TD3Agent` objects and loads their state, and loads replay buffers and RNG into temporary values. Only after it returns successfully may `load` swap `self.defender`, `self.attacker_agents`, and `self.attacker_buffers`; rebind both dictionaries on `best_response` and `task_runner`; and restore Python/NumPy/Torch RNG state.

- [ ] **Step 7: Prove atomic replacement rollback still works**

Patch one buffer's `save` to raise during replacement, then assert the prior `latest/checkpoint.json` iteration is unchanged and no `.latest.staging-*` or `.latest.backup-*` paths remain.

- [ ] **Step 8: Run checkpoint tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py -q -k checkpoint`

Expected: complete-state, mutation guard, CUDA-conditional, and rollback tests pass.

- [ ] **Step 9: Commit complete checkpoint support**

```bash
git add meta_sg/learning/meta_sg_trainer.py meta_sg/tests/test_meta_sg_checkpoint_resume.py
git commit -m "feat: checkpoint complete Meta-SG training state"
```

### Task 5: Make resume explicit and prove deterministic continuity

**Files:**
- Modify: `meta_sg/learning/meta_sg_trainer.py`
- Modify: `meta_sg/scripts/run_meta_sg_pretraining.py`
- Modify: `meta_sg/tests/test_meta_sg_checkpoint_resume.py`
- Modify: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Write failing legacy-policy tests**

```python
with pytest.raises(ValueError, match="model-only"):
    trainer.load(legacy_checkpoint)
trainer.load(legacy_checkpoint, allow_model_only=True)
assert trainer.resume_fidelity == "model_only_discontinuity"
```

Also assert `--allow-model-only-resume` defaults false and cannot be supplied without `--resume-from`.

- [ ] **Step 2: Write a failing uninterrupted-versus-resumed stub test**

Run a seeded 4-iteration stub trainer uninterrupted. Separately run two iterations, save schema-2 `latest`, construct a fresh trainer without reseeding over the saved state, load, and run iterations 3–4. Compare the iteration-3/4 JSON fields that define learning behavior:

```python
for expected, actual in zip(uninterrupted_records[2:], resumed_records):
    assert actual["batch_attack_types"] == expected["batch_attack_types"]
    assert actual["reward_mean"] == pytest.approx(expected["reward_mean"], abs=1e-12)
    assert actual["reptile_delta_norm"] == pytest.approx(expected["reptile_delta_norm"], abs=1e-12)
    assert actual["reptile_actor_delta_norm"] == pytest.approx(expected["reptile_actor_delta_norm"], abs=1e-12)
    assert actual["reptile_critic_delta_norm"] == pytest.approx(expected["reptile_critic_delta_norm"], abs=1e-12)
    assert actual["buffer_sizes"] == expected["buffer_sizes"]
```

Exclude wall-time and memory-maintenance telemetry from equality.

- [ ] **Step 3: Run resume tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py meta_sg/tests/test_meta_sg_pretraining.py -q -k 'legacy or deterministic_resume or can_resume'`

Expected: legacy resume is currently silent and deterministic continuation diverges.

- [ ] **Step 4: Seed and restore every supported process RNG**

At startup add:

```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)
```

Load the checkpoint only after trainer construction; the loaded RNG state must overwrite random draws consumed during construction/probing.

- [ ] **Step 5: Enforce legacy opt-in and expose fidelity**

Add `allow_model_only: bool = False` to `MetaSGTrainer.load`. Schema-less checkpoints raise unless true. When true, load only agent files, set `resume_fidelity = "model_only_discontinuity"`, and print one bounded warning. Schema-2 loads set `resume_fidelity = "continuous"`.

Add CLI validation:

```python
args = parser.parse_args(argv)
if args.allow_model_only_resume and not args.resume_from:
    parser.error("--allow-model-only-resume requires --resume-from")
return args
```

- [ ] **Step 6: Run deterministic resume coverage**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_checkpoint_resume.py meta_sg/tests/test_meta_sg_pretraining.py -q -k 'legacy or deterministic_resume or can_resume'`

Expected: all selected tests pass; schema-2 continuation matches uninterrupted learning metrics.

- [ ] **Step 7: Commit recovery policy**

```bash
git add meta_sg/learning/meta_sg_trainer.py meta_sg/scripts/run_meta_sg_pretraining.py meta_sg/tests/test_meta_sg_checkpoint_resume.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: enforce continuous Meta-SG resume fidelity"
```

### Task 6: Preserve per-attempt configuration and provenance

**Files:**
- Modify: `meta_sg/scripts/experiment_artifacts.py`
- Modify: `meta_sg/scripts/run_meta_sg_pretraining.py`
- Modify: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Write failing immutable-provenance and attempt tests**

Invoke `provenance` twice with different configuration and assert the first payload remains byte-identical. Invoke a new `attempt` command twice and assert `attempts.jsonl` has two strict-JSON lines with `attempt_id`, `started_at`, `start_iteration`, `reason`, `command`, `config_sha256`, `resume_fidelity`, and later completion fields `finished_at` and `exit_code`.

- [ ] **Step 2: Write a failing config-history test**

Run the pretraining CLI twice into the same explicit run name and assert:

```python
assert (run_dir / "config.json").read_bytes() == initial_config_bytes
attempt_configs = sorted((run_dir / "attempts").glob("*/config.json"))
assert len(attempt_configs) == 2
assert json.loads(attempt_configs[1].read_text())["start_iteration"] == 1
```

- [ ] **Step 3: Run artifact tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q -k 'provenance or attempt or config_history'`

Expected: FAIL because provenance overwrites, no attempt command exists, and config is replaced.

- [ ] **Step 4: Add immutable and append-only helper operations**

Make `write_provenance` return without mutation when its output exists. Add an `attempt` subcommand using a single append-mode write plus `flush`/`os.fsync`; derive `config_sha256` from exact config bytes and write lifecycle updates as separate records sharing the same attempt ID.

The new parser contract is:

```text
experiment_artifacts.py attempt --output attempts.jsonl --attempt-id ID
  --phase started|finished --start-iteration N --reason TEXT
  --command-json JSON --config PATH --resume-fidelity fresh|continuous|model_only_discontinuity
  [--exit-code N]
```

- [ ] **Step 5: Write initial and attempt-specific configs without replacement**

Use `attempt_id = UTC timestamp + PID`, write `attempts/<attempt_id>/config.json`, and create root `config.json` with exclusive mode `x` only when absent. Include `attempt_id` and `resume_fidelity` in the attempt-specific record.

- [ ] **Step 6: Run artifact tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q -k 'provenance or attempt or config_history or status_preserves'`

Expected: all selected tests pass; initial provenance/config bytes remain unchanged.

- [ ] **Step 7: Commit append-only artifacts**

```bash
git add meta_sg/scripts/experiment_artifacts.py meta_sg/scripts/run_meta_sg_pretraining.py meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "feat: preserve Meta-SG run attempt history"
```

### Task 7: Make the H=200 launcher memory-safe and resume-safe

**Files:**
- Modify: `meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh`
- Modify: `meta_sg/tests/test_meta_sg_pretraining.py`

- [ ] **Step 1: Strengthen the failing launcher contract test**

```python
assert 'MEMORY_MAINTENANCE="${MEMORY_MAINTENANCE:-task}"' in text
assert '--memory-maintenance "${MEMORY_MAINTENANCE}"' in text
assert 'if [[ ! -f "${RESOURCE_CSV}" ]]' in text
assert 'attempts.jsonl' in text
assert '--phase started' in text
assert '--phase finished' in text
assert 'ALLOW_MODEL_ONLY_RESUME="${ALLOW_MODEL_ONLY_RESUME:-0}"' in text
assert '--allow-model-only-resume' in text
```

- [ ] **Step 2: Add a launcher stub resume integration test**

Run the launcher twice with one stub iteration per attempt and the second invocation pointed at schema-2 `latest`. Assert resource CSV contains exactly one header, metrics has global iterations `[1, 2]`, root provenance is unchanged, two attempt configs exist, attempt lifecycle records append, and evaluation runs only on the attempt whose `START_ITERATION + T == TOTAL_ITERATIONS`.

- [ ] **Step 3: Run launcher tests and verify failure**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q -k 'global_model_poisoning_h200_launcher'`

Expected: FAIL because the launcher truncates CSV and lacks memory/attempt flags.

- [ ] **Step 4: Add launcher defaults and append semantics**

Add:

```bash
MEMORY_MAINTENANCE="${MEMORY_MAINTENANCE:-task}"
ALLOW_MODEL_ONLY_RESUME="${ALLOW_MODEL_ONLY_RESUME:-0}"
ATTEMPTS_JSONL="${RUN_DIR}/attempts.jsonl"
ATTEMPT_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
if [[ -n "${RESUME_FROM}" ]]; then
  ATTEMPT_REASON="${ATTEMPT_REASON:-resume}"
else
  ATTEMPT_REASON="${ATTEMPT_REASON:-fresh}"
fi
```

Create the resource header only when the file is absent or empty. Keep `tee -a`. Create provenance only when absent, and append started/finished attempt records on every launcher exit path.

- [ ] **Step 5: Wire maintenance and explicit legacy policy**

Add `--memory-maintenance "${MEMORY_MAINTENANCE}"` to training. Append `--allow-model-only-resume` only when `ALLOW_MODEL_ONLY_RESUME=1`; the new formal run must leave it at `0`.

- [ ] **Step 6: Guard final evaluation by completed global iteration**

After training, read lightweight `checkpoint.json` metadata only. If completed iteration is below `TOTAL_ITERATIONS`, record a paused/partial attempt and exit successfully without final evaluation. Invoke clean/IPM/LMP/RL evaluation only when the completed iteration equals 100 for the formal run.

- [ ] **Step 7: Run launcher integration tests**

Run: `.venv/bin/python -m pytest meta_sg/tests/test_meta_sg_pretraining.py -q -k 'global_model_poisoning_h200_launcher or attempt or status_preserves'`

Expected: all selected tests pass and no existing artifact is truncated.

- [ ] **Step 8: Commit launcher behavior**

```bash
git add meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh meta_sg/tests/test_meta_sg_pretraining.py
git commit -m "fix: make H200 launcher memory and resume safe"
```

### Task 8: Verify software and gate the new formal run

**Files:**
- Verify: all files changed in Tasks 1–7
- Preserve: `fl_sandbox/runs/meta_sg_global_h200_t100_k10_c30_a6/paper-h200-t100-k10-seed42-20260714T0122Z`

- [ ] **Step 1: Run formatting/static sanity**

Run: `.venv/bin/python -m compileall -q meta_sg`

Expected: exit code 0.

Run: `bash -n meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh`

Expected: exit code 0.

- [ ] **Step 2: Run the complete Meta-SG test suite**

Run: `.venv/bin/python -m pytest meta_sg/tests -q`

Expected: all tests pass with no failures.

- [ ] **Step 3: Run matched one-task real-backend baseline and fix smokes**

Use fresh run IDs and keep all other variables identical:

```bash
RUN_ID=memory-baseline-seed42 BACKEND=fl_sandbox DEVICE=cuda:0 T=1 K=1 H=20 L=2 N_A=2 TOTAL_ITERATIONS=1 MEMORY_MAINTENANCE=off CHECKPOINT_INTERVAL=1 bash meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
RUN_ID=memory-task-seed42 BACKEND=fl_sandbox DEVICE=cuda:0 T=1 K=1 H=20 L=2 N_A=2 TOTAL_ITERATIONS=1 MEMORY_MAINTENANCE=task CHECKPOINT_INTERVAL=1 bash meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
```

Expected: both runs finish, strict JSON metrics are finite, and the maintenance run contains one task maintenance record without changing attack/task configuration.

- [ ] **Step 4: Start a three-iteration formal-configuration gate run**

```bash
RUN_ID=paper-h200-memory-gate-seed42 BACKEND=fl_sandbox DEVICE=cuda:0 T=3 K=10 H=200 L=10 N_A=10 TOTAL_ITERATIONS=3 MEMORY_MAINTENANCE=task CHECKPOINT_INTERVAL=1 MASTER_SEED=42 TRAINING_SEED=42 bash meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
```

Expected: three complete outer records, no swap growth, no non-finite metric, and normal task duration.

- [ ] **Step 5: Compute the lightweight RSS gate without reading checkpoint payloads**

Read only `metrics.jsonl`, `resource_metrics.csv`, logs, status, and `checkpoint.json`. Calculate post-maintenance RSS from the last task record per iteration and fit `numpy.polyfit([1,2,3], rss_gib, 1)[0]`.

Expected: slope is at most `0.2 GiB/outer iteration`; maintenance warnings are absent or explicitly benign; swap remains zero. If it fails, repeat only the gate with `MALLOC_ARENA_MAX=2`. If that fails, repeat only the gate with `FL_PARALLEL_CLIENTS=1`.

- [ ] **Step 6: Start the new seed-42 paper run from iteration zero**

Use a new immutable run ID; do not resume the model-only diagnostic attempt:

```bash
RUN_ID=paper-h200-t100-k10-seed42-memorysafe-20260714 BACKEND=fl_sandbox DEVICE=cuda:0 T=100 K=10 H=200 L=10 N_A=10 TOTAL_ITERATIONS=100 MEMORY_MAINTENANCE=task CHECKPOINT_INTERVAL=10 MASTER_SEED=42 TRAINING_SEED=42 EVALUATION_SEED=10042 bash meta_sg/scripts/run_global_model_poisoning_h200_30c6a.sh
```

Expected: one atomically replaced `checkpoints/latest` every ten iterations, bounded RSS, append-only observability, then clean/IPM/LMP/RL final evaluation after iteration 100.

- [ ] **Step 7: Audit final lightweight results**

Validate strict JSON and finite values in 100 training records and all evaluation scenarios; report reward/query/reptile trends, per-attack final metrics, resource maxima/slope, checkpoint metadata filenames and completed iteration, provenance/attempt continuity, and test evidence. Do not deserialize checkpoint payloads for reporting.

- [ ] **Step 8: Commit any verification-only documentation update**

```bash
git status --short
```

Expected: only deliberately preserved user files remain untracked; no generated run artifacts are staged.
