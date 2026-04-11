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

## Initial Folder Layout

- `attacks.py`
  Standalone attacker implementations rebuilt for isolated validation.
- `context.py`
  Round-level data structures for attacker-only validation.
- `fl_runner.py`
  Minimal MNIST federated-learning runner for attack-side validation.
- `metrics.py`
  Attack-side metrics such as update norm, cosine similarity, clean accuracy,
  and backdoor accuracy.
- `visualize.py`
  Common plotting utilities for attacker validation.
- `run_sandbox.py`
  Entry point for future standalone demos and smoke tests.

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
- standalone `IPM` attacker
- round-wise attack metrics
- basic visualizations for clean accuracy and update norms

If you want a step-by-step run guide for MNIST, see [MNIST_RUN_GUIDE.md](./MNIST_RUN_GUIDE.md).

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
python attacker_sandbox/run_sandbox.py --device auto
python attacker_sandbox/benchmark_clean.py --device cuda:0
```

Both entry points print the resolved runtime device at startup completion so it
is easy to confirm whether the run used GPU.

## Environment Setup

This repo did not previously include a reproducible Python environment file.
Use the helper script below to create a local virtualenv with Python 3.10 and
install the required packages:

```bash
cd /home/antik/rl/Meta_Stackelberg_Learning
bash attacker_sandbox/setup_env.sh cpu
bash attacker_sandbox/setup_env.sh cu121
```

After activation, install mode and device can be checked with:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
