# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project simulates a Bayesian Stackelberg game in Federated Learning (FL) security using Meta-Reinforcement Learning (Meta-RL). A **defender** (leader) learns to dynamically select aggregation rule hyperparameters; **attackers** (followers) perform data poisoning and backdoor attacks.

There are two main workspaces:

- **`src/` + `train_meta_fl.py`** — The full meta-SG experiment stack (meta-RL defender, MAML/REPTILE outer loop). This is intentionally kept separate.
- **`fl_sandbox/`** — A standalone attacker validation sandbox for isolating and verifying attack algorithms before integrating with the full stack. This is the actively developed workspace.

## Commands

### Environment Setup

```bash
# Create virtualenv with CPU-only PyTorch (Python 3.10)
bash fl_sandbox/scripts/setup_env.sh cpu

# With CUDA 12.1 support
bash fl_sandbox/scripts/setup_env.sh cu121

source .venv/bin/activate
```

The project also has a `.venv312` (Python 3.12) venv at the repo root.

### Running Experiments

```bash
# Basic single run (clean FL baseline)
python fl_sandbox/run/run_experiment.py --attack_type clean --device auto

# Attack run with specific defense
python fl_sandbox/run/run_experiment.py --attack_type ipm --defense_type krum --rounds 30

# RL attacker with paper-aligned schedule
python fl_sandbox/run/run_experiment.py \
  --protocol rlfl \
  --attack_type rl \
  --defense_type clipped_median \
  --split_mode paper_q \
  --noniid_q 0.1 \
  --warmup_rounds 100

# Using a YAML config file (CLI args override config values)
python fl_sandbox/run/run_experiment.py --config fl_sandbox/config/run_experiment.example.yaml

# Paper-aligned RL benchmark batch script
python fl_sandbox/scripts/run_rlfl_benchmark.py
```

**Attack types:** `clean`, `ipm`, `lmp`, `bfl`, `dba`, `rl`, `brl`
**Defense types:** `fedavg`, `krum`, `multi_krum`, `median`, `clipped_median`, `trimmed_mean`, `geometric_median`, `fltrust`
**Datasets:** `mnist`, `fmnist`, `cifar10`

### Running Tests

```bash
# All tests
python -m pytest tests/

# Single test file
python -m pytest tests/test_attacker_sandbox_application.py

# Single test case
python -m pytest tests/test_attacker_sandbox_application.py::TestAttackerSandboxApplication::test_split_suffix_formats_modes

# Tests without torch dependency (fast)
python -m pytest tests/test_attacker_sandbox_application.py tests/test_attacker_sandbox_core_defender.py
```

### TensorBoard

```bash
bash fl_sandbox/scripts/run_tensorboard.sh
```

## fl_sandbox Architecture

The sandbox follows a layered design (see `fl_sandbox/docs/OUTER_ARCHITECTURE.md`):

```
fl_sandbox/run/run_experiment.py        ← CLI entry point (thin shell)
    └── fl_sandbox/core/experiment_service.py   ← Orchestrates one experiment run
            ├── experiment_builders.py           ← Factories: build_attack(), build_config()
            ├── fl_runner.py (MinimalFLRunner)   ← Core FL round execution loop
            ├── runtime.py                       ← Shared data structures (RoundContext, RoundSummary, etc.)
            └── postprocess/                     ← TensorBoard/CSV/JSON output helpers
```

### Configuration Flow

`RunConfig` (structured dataclass, `fl_sandbox/config/schema.py`) is the canonical config object. It has sections: `data`, `fl`, `protocol`, `runtime`, `init`, `attacker`, `defender`, `output`.

Config sources, in priority order:
1. YAML file (`--config path.yaml`) loaded via `load_run_config()` → `RunConfig.from_mapping()`
2. CLI overrides merged via `merge_cli_overrides()`
3. `RunConfig()` schema defaults

`RunConfig.normalize()` enforces protocol-specific invariants (e.g., when `protocol=rlfl`, derives RL schedule from `warmup_rounds`).

### Attacker Interface

All attackers extend `SandboxAttack` (`fl_sandbox/core/attacks/base.py`) and implement `execute(ctx: RoundContext, attacker_action) -> List[Weights]`. The `RoundContext` carries everything an attacker needs: old weights, benign client weights, data loaders, device, etc.

Implemented attackers:
- **IPM / LMP** (`attacks/poisoning.py`) — model poisoning
- **BFL / DBA / BRL** (`attacks/backdoor.py`) — backdoor attacks
- **RL** (`attacks/rl.py` + `core/rl/attacker.py`) — paper-style adaptive attacker with TD3 policy, distribution learning, and FL simulator

### Defender Interface

All defenders extend `SandboxDefender` → `AggregationDefender` and expose `aggregate(old_weights, client_weights_list) -> Weights`. The concrete aggregation math lives in `core/defender/aggregation_runtime.py`.

### Round Execution

`MinimalFLRunner.run_many_rounds()` in `fl_runner.py` orchestrates:
1. Client sampling and data partitioning
2. Benign client local training
3. Attacker `observe_round()` then `execute()` to produce malicious weights
4. Defender aggregation
5. Evaluation and `RoundSummary` collection

Output per run: `summary.json`, `client_metrics.csv`, TensorBoard logs under `fl_sandbox/outputs/` and `fl_sandbox/runs/`.

## Key Conventions

- `fl_sandbox/` is fully independent from `src/` and `train_meta_fl.py` — do not import between them (except `fl_runner.py` imports `src/models` and `src/utils` for model/dataset utilities).
- `RunConfig.normalize()` is called after every config construction; ensure it is not bypassed.
- `eval_every` is always forced to `1` by `normalize()` to keep round-wise metrics comparable.
- Run names follow the pattern `{dataset}_{attack}_{defense}_{split_suffix}_{rounds}r`.
- Default output paths: `fl_sandbox/outputs/` and `fl_sandbox/runs/`.
