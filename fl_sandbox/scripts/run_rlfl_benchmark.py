"""Convenience script for one paper-aligned RLFL benchmark run.

This file intentionally lives under ``scripts/`` rather than ``run/``
because it is a specialized preset, not a core runtime surface of the sandbox.

The schedule below follows the NeurIPS 2022 paper defaults much more closely
than the lightweight local benchmarks:
  - 100 total workers / 20 attackers
  - subsampling rate 0.1
  - 1000 FL rounds
  - paper_q non-IID split with q=0.1
  - distribution learning through round 100
  - attack starts at round 101
  - policy learning ends at round 400

To run the newer stealth-aware attacker on the same paper-aligned FL setup:
    python fl_sandbox/scripts/run_rlfl_benchmark.py --attack_type rl2
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.run.run_experiment import main as run_experiment_main


DEFAULT_RLFL_ARGS = [
    "--protocol",
    "rlfl",
    "--attack_type",
    "rl",
    "--rounds",
    "1000",
    "--warmup_rounds",
    "100",
    "--rl_distribution_steps",
    "100",
    "--rl_attack_start_round",
    "101",
    "--rl_policy_train_end_round",
    "400",
    "--num_clients",
    "100",
    "--num_attackers",
    "20",
    "--subsample_rate",
    "0.1",
    "--lr",
    "0.01",
    "--batch_size",
    "128",
    "--eval_batch_size",
    "4096",
    "--num_workers",
    "0",
    "--seed",
    "1001",
    "--defense_type",
    "clipped_median",
    "--split_mode",
    "paper_q",
    "--noniid_q",
    "0.1",
    "--ipm_scaling",
    "5.0",
    "--output_root",
    "fl_sandbox/outputs/rlfl_benchmark",
    "--tb_root",
    "fl_sandbox/runs/rlfl_benchmark",
]


def main(argv: Optional[list[str]] = None) -> None:
    run_experiment_main(
        DEFAULT_RLFL_ARGS + (argv or []),
        description="Run one paper-aligned RLFL benchmark experiment",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
