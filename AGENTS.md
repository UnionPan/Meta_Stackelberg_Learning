# AGENTS.md

## Project Memory

- This repository has a project virtual environment at `.venv/`.
- Use `.venv/bin/python` for project scripts that need PyTorch, TensorBoard, matplotlib, pandas, or TensorFlow-related tooling.
- The system `python3` may not have project dependencies installed.
- To open TensorBoard for the sandbox runs, use:

```bash
TENSORBOARD_HOST=127.0.0.1 TENSORBOARD_PORT=6008 \
  bash fl_sandbox/scripts/run_tensorboard.sh fl_sandbox/runs
```

Equivalent direct command:

```bash
.venv/bin/tensorboard --logdir fl_sandbox/runs --host 127.0.0.1 --port 6008 --load_fast=false
```
