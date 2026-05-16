"""Run paper-scale RL-attacker comparisons across robust defenses.

The script runs, per defense:
  1. clean FL baseline
  2. online RL attacker training
  3. fixed-policy FL evaluation from the saved RL checkpoint
  4. clean-vs-fixed TensorBoard comparison

It intentionally keeps the paper-specific strict reproductions separate from
the canonical multi-defense RL path: clipped-median and Krum use the original
repo formulas, while Median/Trimmed-Mean/FLTrust use the generic defense-aware
TD3 attacker already present in fl_sandbox.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.core.postprocess.tensorboard_utils import build_summary_writer


DEFAULT_CONFIG = Path("fl_sandbox/config/rl_attacker_paper_clipped_median.yaml")
ALL_DEFENSES = (
    "fedavg",
    "krum",
    "multi_krum",
    "median",
    "clipped_median",
    "geometric_median",
    "trimmed_mean",
    "fltrust",
    "paper_norm_trimmed_mean",
)
DEFAULT_DEFENSES = ALL_DEFENSES
SPLIT_SUFFIX = "paper_q_q0.1"


@dataclass(frozen=True)
class DefensePlan:
    name: str
    semantics: str = "canonical"


def defense_plan_for(defense: str) -> DefensePlan:
    if defense == "clipped_median":
        semantics = "legacy_clipped_median_strict"
    elif defense == "krum":
        semantics = "legacy_krum_strict"
    else:
        semantics = "canonical"
    return DefensePlan(name=defense, semantics=semantics)


def run_name_for(attack_type: str, defense_type: str, *, rounds: int) -> str:
    return f"mnist_{attack_type}_{defense_type}_{SPLIT_SUFFIX}_{rounds}r"


def _base_command(
    *,
    config_path: Path,
    attack_type: str,
    defense_type: str,
    output_root: Path,
    tb_root: Path,
    rounds: int,
    distribution_steps: int,
    attack_start_round: int,
    policy_train_end_round: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fl_sandbox.run.run_experiment",
        "--config",
        str(config_path),
        "--attack_type",
        attack_type,
        "--defense_type",
        defense_type,
        "--rounds",
        str(rounds),
        "--rl_distribution_steps",
        str(distribution_steps),
        "--rl_attack_start_round",
        str(attack_start_round),
        "--rl_policy_train_end_round",
        str(policy_train_end_round),
        "--output_root",
        str(output_root),
        "--tb_root",
        str(tb_root),
    ]


def build_clean_command(
    plan: DefensePlan,
    *,
    config_path: Path,
    output_root: Path,
    tb_root: Path,
    rounds: int,
    distribution_steps: int,
    attack_start_round: int,
    policy_train_end_round: int,
) -> list[str]:
    return _base_command(
        config_path=config_path,
        attack_type="clean",
        defense_type=plan.name,
        output_root=output_root,
        tb_root=tb_root,
        rounds=rounds,
        distribution_steps=distribution_steps,
        attack_start_round=attack_start_round,
        policy_train_end_round=policy_train_end_round,
    )


def build_train_command(
    plan: DefensePlan,
    *,
    config_path: Path,
    output_root: Path,
    tb_root: Path,
    rounds: int,
    distribution_steps: int,
    attack_start_round: int,
    policy_train_end_round: int,
    policy_train_steps_per_round: int,
    simulator_horizon: int,
    checkpoint_interval: int,
) -> list[str]:
    cmd = _base_command(
        config_path=config_path,
        attack_type="rl",
        defense_type=plan.name,
        output_root=output_root,
        tb_root=tb_root,
        rounds=rounds,
        distribution_steps=distribution_steps,
        attack_start_round=attack_start_round,
        policy_train_end_round=policy_train_end_round,
    )
    cmd.extend(
        [
            "--rl_attacker_semantics",
            plan.semantics,
            "--rl_policy_train_steps_per_round",
            str(policy_train_steps_per_round),
            "--rl_simulator_horizon",
            str(simulator_horizon),
            "--rl_checkpoint_interval",
            str(checkpoint_interval),
        ]
    )
    return cmd


def build_fixed_eval_command(
    plan: DefensePlan,
    *,
    config_path: Path,
    output_root: Path,
    tb_root: Path,
    policy_checkpoint: Path,
    rounds: int,
    distribution_steps: int,
    attack_start_round: int,
    policy_train_steps_per_round: int,
    simulator_horizon: int,
    checkpoint_interval: int,
) -> list[str]:
    cmd = _base_command(
        config_path=config_path,
        attack_type="rl",
        defense_type=plan.name,
        output_root=output_root,
        tb_root=tb_root,
        rounds=rounds,
        distribution_steps=distribution_steps,
        attack_start_round=attack_start_round,
        policy_train_end_round=0,
    )
    cmd.extend(
        [
            "--rl_attacker_semantics",
            plan.semantics,
            "--rl_policy_train_steps_per_round",
            str(policy_train_steps_per_round),
            "--rl_simulator_horizon",
            str(simulator_horizon),
            "--rl_policy_checkpoint_path",
            str(policy_checkpoint),
            "--rl_freeze_policy",
            "--rl_checkpoint_interval",
            str(checkpoint_interval),
        ]
    )
    return cmd


def _run_command(cmd: Sequence[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write("+ " + " ".join(cmd) + "\n")
        log_fh.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log_fh, stderr=subprocess.STDOUT, check=True)


def _summary_exists(run_dir: Path) -> bool:
    return (run_dir / "summary.json").is_file()


def _postprocess_compare(*, clean_dir: Path, eval_dir: Path, tb_dir: Path, log_path: Path) -> None:
    cmd = [
        sys.executable,
        "fl_sandbox/core/postprocess/postprocess.py",
        "--clean_input_dir",
        str(clean_dir),
        "--attack_input_dir",
        str(eval_dir),
        "--tb_dir",
        str(tb_dir),
    ]
    _run_command(cmd, log_path=log_path)


def numeric_series_from_payload(payload: dict) -> dict[str, list[float]]:
    if "series" in payload:
        series: dict[str, list[float]] = {}
        for key, values in payload["series"].items():
            numeric_values: list[float] = []
            for value in values:
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    numeric_values = []
                    break
            if numeric_values:
                series[key] = numeric_values
        return series
    rounds = payload.get("rounds", [])
    return {
        "clean_acc": [float(row.get("accuracy", row.get("clean_acc", 0.0))) for row in rounds],
        "clean_loss": [float(row.get("loss", row.get("clean_loss", 0.0))) for row in rounds],
    }


def _series_from_summary(summary_path: Path) -> dict[str, list[float]]:
    with summary_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return numeric_series_from_payload(payload)


def _mean_tail(values: Sequence[float], n: int) -> float:
    if not values:
        return 0.0
    tail = list(values[-min(n, len(values)) :])
    return float(sum(tail) / len(tail))


def _write_suite_summary(rows: list[dict[str, float | str]], *, csv_path: Path, tb_dir: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "defense",
        "clean_final_acc",
        "fixed_final_acc",
        "final_acc_drop",
        "clean_last100_acc",
        "fixed_last100_acc",
        "last100_acc_drop",
        "fixed_final_loss",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    writer = build_summary_writer(tb_dir)
    for step, row in enumerate(rows):
        defense = str(row["defense"])
        for key in fieldnames[1:]:
            writer.add_scalar(f"defense/{defense}/{key}", float(row[key]), step)
        writer.add_text(f"defense/{defense}/name", defense, step)
    writer.flush()
    writer.close()


def _completed_row(defense: str, *, clean_dir: Path, eval_dir: Path) -> dict[str, float | str]:
    clean = _series_from_summary(clean_dir / "summary.json")
    fixed = _series_from_summary(eval_dir / "summary.json")
    clean_acc = clean.get("clean_acc", [])
    fixed_acc = fixed.get("clean_acc", [])
    fixed_loss = fixed.get("clean_loss", [])
    return {
        "defense": defense,
        "clean_final_acc": clean_acc[-1] if clean_acc else 0.0,
        "fixed_final_acc": fixed_acc[-1] if fixed_acc else 0.0,
        "final_acc_drop": (clean_acc[-1] - fixed_acc[-1]) if clean_acc and fixed_acc else 0.0,
        "clean_last100_acc": _mean_tail(clean_acc, 100),
        "fixed_last100_acc": _mean_tail(fixed_acc, 100),
        "last100_acc_drop": _mean_tail(clean_acc, 100) - _mean_tail(fixed_acc, 100),
        "fixed_final_loss": fixed_loss[-1] if fixed_loss else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--defenses", nargs="+", default=list(DEFAULT_DEFENSES))
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--distribution-steps", type=int, default=50)
    parser.add_argument("--attack-start-round", type=int, default=51)
    parser.add_argument("--policy-train-end-round", type=int, default=200)
    parser.add_argument("--policy-train-steps-per-round", type=int, default=10)
    parser.add_argument("--simulator-horizon", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--eval-checkpoint-interval", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("fl_sandbox/outputs/paper_defense_suite"))
    parser.add_argument("--tb-root", type=Path, default=Path("fl_sandbox/runs/paper_defense_suite"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    clean_root = args.output_root / "clean"
    train_root = args.output_root / "train"
    eval_root = args.output_root / "fixed_eval"
    log_root = args.output_root / "logs"
    compare_root = args.tb_root / "compare"
    rows: list[dict[str, float | str]] = []

    for defense in args.defenses:
        plan = defense_plan_for(defense)
        clean_dir = clean_root / run_name_for("clean", defense, rounds=args.rounds)
        train_dir = train_root / run_name_for("rl", defense, rounds=args.rounds)
        eval_dir = eval_root / run_name_for("rl", defense, rounds=args.rounds)
        checkpoint = train_dir / "checkpoints" / "rl_policy_latest.pt"
        log_path = log_root / f"{defense}.log"

        if args.force or not _summary_exists(clean_dir):
            _run_command(
                build_clean_command(
                    plan,
                    config_path=args.config,
                    output_root=clean_root,
                    tb_root=args.tb_root / "clean",
                    rounds=args.rounds,
                    distribution_steps=args.distribution_steps,
                    attack_start_round=args.attack_start_round,
                    policy_train_end_round=args.policy_train_end_round,
                ),
                log_path=log_path,
            )

        if args.force or not _summary_exists(train_dir) or not checkpoint.is_file():
            _run_command(
                build_train_command(
                    plan,
                    config_path=args.config,
                    output_root=train_root,
                    tb_root=args.tb_root / "train",
                    rounds=args.rounds,
                    distribution_steps=args.distribution_steps,
                    attack_start_round=args.attack_start_round,
                    policy_train_end_round=args.policy_train_end_round,
                    policy_train_steps_per_round=args.policy_train_steps_per_round,
                    simulator_horizon=args.simulator_horizon,
                    checkpoint_interval=args.checkpoint_interval,
                ),
                log_path=log_path,
            )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing RL policy checkpoint for {defense}: {checkpoint}")

        if args.force or not _summary_exists(eval_dir):
            _run_command(
                build_fixed_eval_command(
                    plan,
                    config_path=args.config,
                    output_root=eval_root,
                    tb_root=args.tb_root / "fixed_eval",
                    policy_checkpoint=checkpoint,
                    rounds=args.rounds,
                    distribution_steps=args.distribution_steps,
                    attack_start_round=args.attack_start_round,
                    policy_train_steps_per_round=args.policy_train_steps_per_round,
                    simulator_horizon=args.simulator_horizon,
                    checkpoint_interval=args.eval_checkpoint_interval,
                ),
                log_path=log_path,
            )

        _postprocess_compare(
            clean_dir=clean_dir,
            eval_dir=eval_dir,
            tb_dir=compare_root / defense,
            log_path=log_path,
        )
        rows.append(_completed_row(defense, clean_dir=clean_dir, eval_dir=eval_dir))
        _write_suite_summary(rows, csv_path=args.output_root / "suite_summary.csv", tb_dir=args.tb_root / "suite_summary")

    print(f"Suite summary CSV: {args.output_root / 'suite_summary.csv'}")
    print(f"Suite summary TensorBoard: {args.tb_root / 'suite_summary'}")


if __name__ == "__main__":
    main(sys.argv[1:])
