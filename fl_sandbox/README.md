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

The source tree is organized around public package boundaries. External code
should import attackers from `fl_sandbox.attacks`, defenders from
`fl_sandbox.defenders`, and aggregation rules from `fl_sandbox.aggregators`.

```text
fl_sandbox/
  attacks/                 fixed attacks and the public attack factory
    rl_attacker/           adaptive RL attacker package
      proxy/               proxy buffer, inversion, and distribution learner
      simulator/           Gymnasium-compatible FL simulator and reward logic
      tianshou_backend/    TD3/PPO trainers backed by Tianshou
  defenders/               defender adapters and defender factory
  aggregators/             pure aggregation / robust-defense algorithms
  federation/              client, partitioning, poisoning, and FL runner logic
  config/                  typed config objects, parser, and YAML examples
  data/                    sandbox-local dataset loading and poisoning helpers
  models/                  model registry and state helpers
  evaluation/              evaluation helpers
  application/             batch and experiment-service orchestration layer
  core/                    runtime glue, metrics, builders, and postprocess tools
  run/                     Python entry points for single-run experiments
  apps/                    app-facing wrappers around run entry points
  scripts/                 shell helpers and benchmark presets
  docs/                    usage guides, specs, and implementation plans
  assets/                  static inspection/doc assets
```

Generated local artifacts are intentionally kept out of the source boundary:

```text
fl_sandbox/outputs/        experiment outputs
fl_sandbox/runs/           TensorBoard and run directories
fl_sandbox/logs/           long-running command logs
__pycache__/               Python bytecode caches
```

Those generated paths are ignored by git and can be removed locally when a run
is no longer needed.

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
- adaptive `RL` attacker under `attacks/rl_attacker/`, split into proxy learning,
  simulation, action decoding, diagnostics, and a Tianshou-backed trainer protocol
- TD3 is the default RL algorithm; PPO is selectable with `--rl_algorithm ppo`
- minimal single-agent `gymnasium` attacker RL environment for simulator training
- round-wise attack metrics including backdoor accuracy
- TensorBoard postprocess helpers for clean-vs-attack comparison

If you want a step-by-step run guide for MNIST, see [MNIST_RUN_GUIDE.md](./docs/MNIST_RUN_GUIDE.md).

For the RL attacker architecture and paper-reproduction comparison, see
[rl_attacker_paper_comparison_zh.md](./docs/rl_attacker_paper_comparison_zh.md).

For the clipped-median strict reproduction and scale-aware TD3 run result, see
[rl_attacker_scaleaware_result_2026_05_12.md](./docs/rl_attacker_scaleaware_result_2026_05_12.md).

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
python fl_sandbox/run/run_experiment.py --attack_type clean --device auto
python fl_sandbox/run/run_experiment.py --attack_type ipm --device cuda:0
```

For the paper-style untargeted `RL` attacker, the sandbox exposes the main
online-learning schedule and algorithm directly from CLI:

```bash
python fl_sandbox/run/run_experiment.py \
  --attack_type rl \
  --defense_type krum \
  --rl_algorithm td3 \
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
python fl_sandbox/scripts/run_rlfl_benchmark.py
```

For custom protocol-aligned single runs, prefer:

```bash
python fl_sandbox/run/run_experiment.py \
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
bash fl_sandbox/scripts/setup_env.sh cpu
bash fl_sandbox/scripts/setup_env.sh cu121
```

After activation, install mode and device can be checked with:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
