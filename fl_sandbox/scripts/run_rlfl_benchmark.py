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

The default ``rl`` attacker is the latest stealth-aware TD3 implementation.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.run.run_experiment import main as run_experiment_main


DEFAULT_RLFL_ARGS = ["--config", "fl_sandbox/config/presets/rlfl_paper.yaml"]


def main(argv: Optional[list[str]] = None) -> None:
    run_experiment_main(
        DEFAULT_RLFL_ARGS + (argv or []),
        description="Run one paper-aligned RLFL benchmark experiment",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
