# Attacker Sandbox

This folder is a standalone workspace for validating FL attacker algorithms
without coupling them to the full Meta-SG experiment stack.

## Goals

- isolate attacker logic from the full training pipeline
- validate attacks one by one before reconnecting them to the main experiment
- visualize attacker behavior round by round

## Planned Build Order

1. clean FL round runner on MNIST
2. fixed model-poisoning attacker: `IPM`
3. fixed backdoor attacker: `BFL`
4. distributed backdoor attacker: `DBA`
5. adaptive attackers: `RL`, `BRL`

## Folder Layout

- `context.py`
  Shared sandbox data structures.
- `run/`
  Python run entry points. `run_experiment.py` is the unified single-run
  experiment entry.
- `core/`
  Shared runtime helpers, experiment builders, metrics, and postprocess entry points.
  Attacker implementations do not live under `core/`.
  Visualization/export helpers now live under `core/postprocess/`, including plotting helpers
  (`visualization.py`), shared TensorBoard exporters (`tensorboard_utils.py`), the unified Python
  postprocess entry point (`postprocess.py`), and compatibility wrappers
  (`postprocess_clean.py`, `postprocess_sandbox.py`).
- `attacks/`
  Public attacker subsystem. External code should import attackers and the attack factory from
  `fl_sandbox.attacks`. Fixed attackers live as direct files, and the adaptive RL attacker lives
  under `attacks/rl_attacker/`.
- `scripts/`
  Shell and preset helpers (`setup_env.sh`, `run_tensorboard.sh`,
  `postprocess_mnist_30r_all.sh`, `run_rlfl_benchmark.py`).
- `docs/`
  Usage guides and operational docs.
- `assets/`
  Static image assets used in docs/inspection.
- `config/`
  Run configuration examples for experiment entry points.
- `envs/`
  Environment/dependency files (`requirements.txt`, `environment.yml`).
- `outputs/` / `runs/`
  Experiment outputs and TensorBoard logs.

## Scope

This sandbox is intentionally separate from:

- `train_meta_fl.py`
- `src/envs/fl_meta_env.py`
- `src/algos/meta_sg.py`

Those files remain the full experiment path. This folder is for rebuilding and
verifying the attacker side in a simpler and more testable way first.

## Current Status

The first runnable slice in this folder is:

- MNIST clean FL
- standalone `IPM` / `LMP` untargeted attackers
- standalone `BFL` / `DBA` backdoor attackers
- paper-style legacy TD3 `RL` attacker preserved under `attacks/rl_attacker/` for Phase 1a; the next phase replaces the hand-written TD3 backend with Tianshou SAC/TD3 trainers
- minimal single-agent `gymnasium` attacker RL environment
- round-wise attack metrics including backdoor accuracy
- TensorBoard postprocess helpers for clean-vs-attack comparison

If you want a step-by-step run guide for MNIST, see [MNIST_RUN_GUIDE.md](./docs/MNIST_RUN_GUIDE.md).

## GPU Support

The sandbox now accepts an explicit runtime device:

- `--device auto`
  Prefer CUDA when available, otherwise fall back to CPU.
- `--device cpu`
  Force CPU execution.
- `--device cuda`
- `--device cuda:0`
  Force a CUDA device explicitly.

Example:

```bash
python attacker_sandbox/run/run_experiment.py --attack_type clean --device auto
python attacker_sandbox/run/run_experiment.py --attack_type ipm --device cuda:0
```

For the paper-style untargeted `RL` attacker, the sandbox exposes the main
online-learning schedule directly from CLI:

```bash
python attacker_sandbox/run/run_experiment.py \
  --attack_type rl \
  --defense_type krum \
  --rounds 50 \
  --rl_distribution_steps 10 \
  --rl_attack_start_round 10 \
  --rl_policy_train_end_round 30
```

The sandbox defaults are intentionally smaller than the paper schedule so local
validation stays tractable. Increase the `rl_*` arguments when you want a
closer paper-scale run.

For a more paper-aligned benchmark protocol with fixed warmup rounds and
round-wise `clean_acc` plus `backdoor_acc` / `ASR` CSV outputs, use:

```bash
python attacker_sandbox/scripts/run_rlfl_benchmark.py
```

For custom protocol-aligned single runs, prefer:

```bash
python attacker_sandbox/run/run_experiment.py \
  --protocol rlfl \
  --attack_type rl \
  --defense_type clipped_median \
  --split_mode paper_q \
  --noniid_q 0.1 \
  --warmup_rounds 100
```

Both entry points print the resolved runtime device at startup completion so it
is easy to confirm whether the run used GPU.

## Environment Setup

This repo did not previously include a reproducible Python environment file.
Use the helper script below to create a local virtualenv with Python 3.10 and
install the required packages:

```bash
cd /home/antik/rl/Meta_Stackelberg_Learning
bash attacker_sandbox/scripts/setup_env.sh cpu
bash attacker_sandbox/scripts/setup_env.sh cu121
```

After activation, install mode and device can be checked with:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
