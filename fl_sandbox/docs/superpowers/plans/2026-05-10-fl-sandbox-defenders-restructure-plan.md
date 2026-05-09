# fl_sandbox Defenders Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move defender adapter APIs from `fl_sandbox.core.defender` to `fl_sandbox.defenders` and delete the old core path.

**Architecture:** `fl_sandbox.defenders` becomes the package-level public API for defender wrappers and factories. `fl_sandbox.aggregators` remains the pure algorithm layer. `fl_sandbox.core` keeps runtime/builders/postprocess only and no longer re-exports defender symbols.

**Tech Stack:** Python 3.10+, pytest, existing `fl_sandbox` dataclasses and aggregation runtime.

---

## File Map

Create:

- `fl_sandbox/defenders/__init__.py`
- `fl_sandbox/defenders/base.py`
- `fl_sandbox/defenders/aggregation.py`
- `fl_sandbox/defenders/aggregation_runtime.py`
- `fl_sandbox/defenders/factory.py`
- `tests/test_fl_sandbox_defenders_public_api.py`

Modify:

- `fl_sandbox/core/experiment_builders.py`
- `fl_sandbox/run/run_experiment.py`
- `fl_sandbox/attacks/lmp.py`
- `fl_sandbox/core/__init__.py`
- `fl_sandbox/README.md`
- defender-related tests

Delete:

- `fl_sandbox/core/defender/`

## Task 1: Add Public API and Old-Import Guard Tests

**Files:**
- Create: `tests/test_fl_sandbox_defenders_public_api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fl_sandbox_defenders_public_api.py` with tests that:

- import `DEFENSE_CHOICES`, `SandboxDefender`, typed defender wrappers,
  `create_defender`, `supported_defense_types`, and
  `build_defender_config_kwargs` from `fl_sandbox.defenders`;
- assert `DEFENSE_CHOICES == supported_defense_types()`;
- assert `paper_norm_trimmed_mean` creates `PaperNormTrimmedMeanDefender`;
- assert `build_config(RunConfig.from_flat_dict(...))` accepts
  `paper_norm_trimmed_mean`;
- scan `fl_sandbox`, `meta_sg`, and `tests` Python imports and fail on
  `fl_sandbox.core.defender`, except inside this guard test.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest tests/test_fl_sandbox_defenders_public_api.py -q
```

Expected: failure because `fl_sandbox.defenders` does not exist yet.

- [ ] **Step 3: Commit tests**

```bash
git add tests/test_fl_sandbox_defenders_public_api.py
git commit -m "test: capture defender public api migration"
```

## Task 2: Move Defender Adapter Package

**Files:**
- Move: `fl_sandbox/core/defender/*.py` to `fl_sandbox/defenders/*.py`

- [ ] **Step 1: Move files**

Run:

```bash
mkdir -p fl_sandbox/defenders
git mv fl_sandbox/core/defender/base.py fl_sandbox/defenders/base.py
git mv fl_sandbox/core/defender/aggregation.py fl_sandbox/defenders/aggregation.py
git mv fl_sandbox/core/defender/aggregation_runtime.py fl_sandbox/defenders/aggregation_runtime.py
git mv fl_sandbox/core/defender/factory.py fl_sandbox/defenders/factory.py
git mv fl_sandbox/core/defender/__init__.py fl_sandbox/defenders/__init__.py
```

- [ ] **Step 2: Update package-relative imports**

Update moved files so all relative imports resolve inside
`fl_sandbox.defenders`.

- [ ] **Step 3: Run defender public tests**

Run:

```bash
PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest tests/test_fl_sandbox_defenders_public_api.py -q
```

Expected: public import assertions pass, old-import guard still fails until
Task 3 migrates callers.

- [ ] **Step 4: Commit moved package**

```bash
git add fl_sandbox/defenders fl_sandbox/core/defender tests/test_fl_sandbox_defenders_public_api.py
git commit -m "refactor: move defender adapters to public package"
```

## Task 3: Migrate In-Repo Imports

**Files:**
- Modify importer files found by `rg "fl_sandbox\\.core\\.defender|from \\.defender"`

- [ ] **Step 1: Update production imports**

Use these import targets:

```python
from fl_sandbox.defenders import DEFENSE_CHOICES
from fl_sandbox.defenders import build_defender_config_kwargs
from fl_sandbox.defenders import AggregationDefender
from fl_sandbox.aggregators import median_aggregate
```

Expected known changes:

- `fl_sandbox/run/run_experiment.py`
- `fl_sandbox/core/experiment_builders.py`
- `fl_sandbox/attacks/lmp.py`

- [ ] **Step 2: Update tests**

Change tests importing `fl_sandbox.core.defender` to import
`fl_sandbox.defenders`.

- [ ] **Step 3: Run guard search**

Run:

```bash
rg -n "fl_sandbox\\.core\\.defender|from \\.defender" fl_sandbox meta_sg tests -g '*.py'
```

Expected: only guard-test string literals.

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest \
  tests/test_fl_sandbox_defenders_public_api.py \
  tests/test_attacker_sandbox_core_defender.py \
  tests/test_attacker_sandbox_defenses.py \
  tests/test_rl_attacker.py -q
```

Expected: pass.

- [ ] **Step 5: Commit import migration**

```bash
git add fl_sandbox meta_sg tests
git commit -m "refactor: migrate defender imports"
```

## Task 4: Delete Old Core Defender Path and Core Re-exports

**Files:**
- Delete: `fl_sandbox/core/defender/`
- Modify: `fl_sandbox/core/__init__.py`

- [ ] **Step 1: Remove old path**

Run:

```bash
git rm -r fl_sandbox/core/defender
```

If already empty, remove ignored cache directories with `rm -rf` and `rmdir`.

- [ ] **Step 2: Remove defender lazy exports from `fl_sandbox/core/__init__.py`**

Delete all entries for `DEFENSE_CHOICES`, `AggregationDefender`,
defender aggregate functions, `build_defender_config_kwargs`, and
`create_defender`.

- [ ] **Step 3: Verify old tracked path is gone**

Run:

```bash
git ls-files fl_sandbox/core/defender
```

Expected: no output.

- [ ] **Step 4: Run full tests**

Run:

```bash
PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest -q
```

Expected: pass.

- [ ] **Step 5: Commit deletion**

```bash
git add fl_sandbox/core/__init__.py
git commit -m "refactor: remove core defender path"
```

## Task 5: Docs and Final Verification

**Files:**
- Modify: `fl_sandbox/README.md`

- [ ] **Step 1: Update README layout**

Document `defenders/` as the public defender adapter/factory API and keep
`aggregators/` as pure aggregation algorithms. Ensure `core/` text says defender
implementations do not live under `core/`.

- [ ] **Step 2: Final searches**

Run:

```bash
rg -n "fl_sandbox\\.core\\.defender|core/defender" fl_sandbox meta_sg tests -g '*.py' -g '*.md'
git ls-files fl_sandbox/core/defender
```

Expected: no source/doc matches except plan/spec historical text if the search
is intentionally scoped to docs under `superpowers`.

- [ ] **Step 3: Final tests**

Run:

```bash
PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest -q
```

Expected: pass.

- [ ] **Step 4: Commit docs**

```bash
git add fl_sandbox/README.md
git commit -m "docs: document defender public api"
```

