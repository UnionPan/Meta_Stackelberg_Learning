"""Run a paper-aligned RL2 benchmark in known-defense mode.

This preset is intentionally more attack-centric than the generic RLFL benchmark:
  - fixed/known defense type supplied up front
  - paper-aligned FL environment (100 clients, 20 attackers, q=0.1)
  - policy keeps adapting through the full FL horizon
  - more inner-loop episodes / horizon / inversion budget for stronger attacks

Primary use: estimate the upper-bound attack effect of RL2 when the defense type is known.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.run.run_experiment import main as run_experiment_main


DEFAULT_ARGS = [
    "--protocol",
    "rlfl",
    "--attack_type",
    "rl2",
    "--rounds",
    "1000",
    "--warmup_rounds",
    "100",
    "--rl_distribution_steps",
    "100",
    "--rl_attack_start_round",
    "101",
    "--rl_policy_train_end_round",
    "1000",
    "--rl_policy_train_episodes_per_round",
    "4",
    "--rl_simulator_horizon",
    "12",
    "--rl_inversion_steps",
    "100",
    "--rl_reconstruction_batch_size",
    "16",
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
    "--output_root",
    "fl_sandbox/outputs/rl2_known_defense",
    "--tb_root",
    "fl_sandbox/runs/rl2_known_defense",
]


def main(argv: Optional[list[str]] = None) -> None:
    run_experiment_main(
        DEFAULT_ARGS + (argv or []),
        description="Run a paper-style RL2 attack with known fixed defense",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
