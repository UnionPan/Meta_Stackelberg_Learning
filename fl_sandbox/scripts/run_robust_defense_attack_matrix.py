"""Run attack benchmarks against Clipped Median and Krum.

The matrix keeps defense-specific RL semantics separated by output directory,
so strict paper reproductions, optimized RL variants, and fixed attack
benchmarks can be compared without run-name collisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.core.postprocess.tensorboard_utils import build_summary_writer


DEFAULT_CONFIG = Path("fl_sandbox/config/rl_attacker_paper_clipped_median.yaml")
DEFAULT_DEFENSES = ("clipped_median", "krum")
SPLIT_SUFFIX = "paper_q_q0.1"
BENCHMARK_ATTACK_NAMES = (
    "clean",
    "ipm",
    "lmp",
    "dba",
    "bfl",
)
OPTIMIZED_RL_ATTACK_NAMES = (
    "rl_clipped_median_scaleaware",
    "rl_krum_geometry",
)
PAPER_RL_ATTACK_NAMES = ("rl_clipped_median_strict", "rl_krum_strict")
HEURISTIC_ATTACK_NAMES = ("clipped_median_geometry_search", "krum_geometry_search")
DEFAULT_ATTACK_NAMES = BENCHMARK_ATTACK_NAMES + OPTIMIZED_RL_ATTACK_NAMES


@dataclass(frozen=True)
class AttackPlan:
    name: str
    attack_type: str
    benchmark: str
    defenses: tuple[str, ...] = DEFAULT_DEFENSES
    rl_semantics: str = ""
    extra_args: tuple[str, ...] = ()


ALL_ATTACK_PLANS = (
    AttackPlan("clean", "clean", "baseline"),
    AttackPlan("ipm", "ipm", "benchmark"),
    AttackPlan("lmp", "lmp", "benchmark"),
    AttackPlan("dba", "dba", "benchmark", extra_args=("--dba_poison_frac", "0.5", "--dba_num_sub_triggers", "4")),
    AttackPlan("bfl", "bfl", "benchmark", extra_args=("--bfl_poison_frac", "1.0")),
    AttackPlan("alie", "alie", "benchmark"),
    AttackPlan("signflip", "signflip", "benchmark"),
    AttackPlan("gaussian", "gaussian", "benchmark"),
    AttackPlan(
        "clipped_median_geometry_search",
        "clipped_median_geometry_search",
        "heuristic",
        defenses=("clipped_median",),
    ),
    AttackPlan("krum_geometry_search", "krum_geometry_search", "heuristic", defenses=("krum",)),
    AttackPlan(
        "rl_clipped_median_strict",
        "rl",
        "rl",
        defenses=("clipped_median",),
        rl_semantics="legacy_clipped_median_strict",
    ),
    AttackPlan(
        "rl_clipped_median_scaleaware",
        "rl",
        "rl",
        defenses=("clipped_median",),
        rl_semantics="legacy_clipped_median_scaleaware",
    ),
    AttackPlan(
        "rl_krum_strict",
        "rl",
        "rl",
        defenses=("krum",),
        rl_semantics="legacy_krum_strict",
    ),
    AttackPlan(
        "rl_krum_geometry",
        "rl",
        "rl",
        defenses=("krum",),
        rl_semantics="legacy_krum_geometry",
    ),
)


def attack_plan_by_name(name: str) -> AttackPlan:
    for plan in ALL_ATTACK_PLANS:
        if plan.name == name:
            return plan
    raise KeyError(f"Unknown attack plan: {name}")


def attack_plans_for_defense(defense: str, requested: Sequence[str] | None = None) -> list[AttackPlan]:
    requested_set = set(requested or DEFAULT_ATTACK_NAMES)
    unknown = requested_set - {plan.name for plan in ALL_ATTACK_PLANS}
    if unknown:
        raise KeyError(f"Unknown attack plan(s): {', '.join(sorted(unknown))}")
    plans = [
        plan
        for plan in ALL_ATTACK_PLANS
        if plan.name in requested_set and defense in plan.defenses
    ]
    if "clean" not in {plan.name for plan in plans}:
        plans.insert(0, attack_plan_by_name("clean"))
    return plans


def parse_reuse_summary_specs(specs: Sequence[str] | None) -> dict[tuple[str, str], Path]:
    mapping: dict[tuple[str, str], Path] = {}
    for spec in specs or []:
        parts = str(spec).split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                "Reuse summary specs must be formatted as defense:attack_name:/path/to/run_dir_or_summary.json"
            )
        defense, attack_name, path = parts
        mapping[(defense, attack_name)] = Path(path)
    return mapping


def truncate_payload_rounds(payload: dict, rounds: int) -> dict:
    rounds = int(rounds)
    if rounds <= 0:
        return dict(payload)
    limited = dict(payload)
    if isinstance(payload.get("series"), dict):
        limited["series"] = {
            key: values[:rounds] if isinstance(values, list) else values
            for key, values in payload["series"].items()
        }
    if isinstance(payload.get("rounds"), list):
        limited["rounds"] = payload["rounds"][:rounds]
    return limited


def summary_payload_round_count(payload: dict) -> int:
    series = payload.get("series")
    if isinstance(series, dict):
        for values in series.values():
            if not isinstance(values, list):
                continue
            numeric_count = 0
            for value in values:
                try:
                    float(value)
                except (TypeError, ValueError):
                    numeric_count = 0
                    break
                numeric_count += 1
            if numeric_count > 0:
                return numeric_count
    rounds = payload.get("rounds")
    return len(rounds) if isinstance(rounds, list) else 0


def run_name_for(attack_name: str, defense: str, *, rounds: int) -> str:
    plan = attack_plan_by_name(attack_name)
    leaf = f"mnist_{plan.attack_type}_{defense}_{SPLIT_SUFFIX}_{rounds}r"
    return f"{defense}/{plan.name}/{leaf}"


def expected_run_dir(root: Path, plan: AttackPlan, defense: str, *, rounds: int) -> Path:
    return root / run_name_for(plan.name, defense, rounds=rounds)


def build_attack_command(
    plan: AttackPlan,
    *,
    defense: str,
    config_path: Path,
    output_root: Path,
    tb_root: Path,
    rounds: int,
    num_clients: int = 100,
    num_attackers: int = 20,
    subsample_rate: float = 0.1,
    krum_attackers: int = 20,
    distribution_steps: int = 100,
    attack_start_round: int = 101,
    policy_train_end_round: int = 0,
    policy_train_steps_per_round: int = 50,
    simulator_horizon: int = 1000,
    checkpoint_interval: int = 25,
    parallel_clients: int | None = 1,
    max_client_samples_per_client: int | None = None,
    max_eval_samples: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fl_sandbox.run.run_experiment",
        "--config",
        str(config_path),
        "--attack_type",
        plan.attack_type,
        "--defense_type",
        defense,
        "--rounds",
        str(rounds),
        "--num_clients",
        str(num_clients),
        "--num_attackers",
        str(num_attackers),
        "--subsample_rate",
        str(subsample_rate),
        "--krum_attackers",
        str(krum_attackers),
        "--rl_distribution_steps",
        str(distribution_steps),
        "--rl_attack_start_round",
        str(attack_start_round),
        "--rl_policy_train_end_round",
        str(policy_train_end_round),
        "--output_root",
        str(output_root / defense / plan.name),
        "--tb_root",
        str(tb_root / defense / plan.name),
    ]
    if parallel_clients is not None:
        cmd.extend(["--parallel_clients", str(parallel_clients)])
    if max_client_samples_per_client is not None:
        cmd.extend(["--max_client_samples_per_client", str(max_client_samples_per_client)])
    if max_eval_samples is not None:
        cmd.extend(["--max_eval_samples", str(max_eval_samples)])
    if plan.rl_semantics:
        cmd.extend(
            [
                "--rl_attacker_semantics",
                plan.rl_semantics,
                "--rl_policy_train_steps_per_round",
                str(policy_train_steps_per_round),
                "--rl_simulator_horizon",
                str(simulator_horizon),
                "--rl_checkpoint_interval",
                str(checkpoint_interval),
                "--rl_save_final_checkpoint",
            ]
        )
    cmd.extend(plan.extra_args)
    return cmd


def _run_command(cmd: Sequence[str], *, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write("+ " + " ".join(cmd) + "\n")
        log_fh.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log_fh, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def _load_payload_from_path(path: Path, *, round_limit: int) -> dict:
    summary_path = path if path.name == "summary.json" else path / "summary.json"
    with summary_path.open(encoding="utf-8") as fh:
        return truncate_payload_rounds(json.load(fh), round_limit)


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


def _mean_tail(values: Sequence[float], n: int) -> float:
    if not values:
        return 0.0
    tail = list(values[-min(int(n), len(values)) :])
    return float(sum(tail) / len(tail))


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def completed_row(
    *,
    defense: str,
    attack_name: str,
    benchmark: str,
    clean_payload: dict,
    attack_payload: dict,
    tail_rounds: int,
    elapsed_s: float = 0.0,
) -> dict[str, float | str]:
    clean = numeric_series_from_payload(clean_payload)
    attack = numeric_series_from_payload(attack_payload)
    clean_acc = clean.get("clean_acc", [])
    attack_acc = attack.get("clean_acc", [])
    attack_loss = attack.get("clean_loss", [])
    clean_tail = _mean_tail(clean_acc, tail_rounds)
    attack_tail = _mean_tail(attack_acc, tail_rounds)
    clean_final = clean_acc[-1] if clean_acc else 0.0
    attack_final = attack_acc[-1] if attack_acc else 0.0
    return {
        "defense": defense,
        "attack_name": attack_name,
        "benchmark": benchmark,
        "clean_final_acc": _round_metric(clean_final),
        "attack_final_acc": _round_metric(attack_final),
        "final_acc_drop": _round_metric(clean_final - attack_final),
        "clean_tail_acc": _round_metric(clean_tail),
        "attack_tail_acc": _round_metric(attack_tail),
        "tail_acc_drop": _round_metric(clean_tail - attack_tail),
        "attack_final_loss": _round_metric(attack_loss[-1] if attack_loss else 0.0),
        "elapsed_s": _round_metric(elapsed_s),
    }


def _write_trend_scalars(
    writer,
    *,
    defense: str,
    attack_name: str,
    clean_payload: dict,
    attack_payload: dict,
) -> None:
    clean = numeric_series_from_payload(clean_payload)
    attack = numeric_series_from_payload(attack_payload)
    for metric, values in attack.items():
        for idx, value in enumerate(values, start=1):
            writer.add_scalar(f"matrix/{defense}/{attack_name}/{metric}", float(value), idx)
    clean_acc = clean.get("clean_acc", [])
    attack_acc = attack.get("clean_acc", [])
    for idx, (clean_value, attack_value) in enumerate(zip(clean_acc, attack_acc), start=1):
        writer.add_scalar(f"matrix/{defense}/{attack_name}/clean_relative_acc_drop", clean_value - attack_value, idx)


def _write_summary_scalars(writer, row: dict[str, float | str], *, step: int) -> None:
    defense = str(row["defense"])
    attack_name = str(row["attack_name"])
    for key in (
        "clean_final_acc",
        "attack_final_acc",
        "final_acc_drop",
        "clean_tail_acc",
        "attack_tail_acc",
        "tail_acc_drop",
        "attack_final_loss",
        "elapsed_s",
    ):
        writer.add_scalar(f"matrix_summary/{defense}/{attack_name}/{key}", float(row[key]), step)
    writer.add_text(f"matrix_summary/{defense}/{attack_name}/benchmark", str(row["benchmark"]), step)


def _write_matrix_summary_csv(rows: list[dict[str, float | str]], *, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "defense",
        "attack_name",
        "benchmark",
        "clean_final_acc",
        "attack_final_acc",
        "final_acc_drop",
        "clean_tail_acc",
        "attack_tail_acc",
        "tail_acc_drop",
        "attack_final_loss",
        "elapsed_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_complete(run_dir: Path, *, rounds: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return summary_payload_round_count(payload) >= int(rounds)


def run_matrix(args: argparse.Namespace) -> None:
    rows: list[dict[str, float | str]] = []
    trend_writer = build_summary_writer(args.tb_root / "matrix_trends")
    summary_writer = build_summary_writer(args.tb_root / "matrix_summary")
    reuse_summaries = parse_reuse_summary_specs(args.reuse_summary)
    for defense in args.defenses:
        plans = attack_plans_for_defense(defense, args.attacks)
        clean_plan = attack_plan_by_name("clean")
        clean_dir = expected_run_dir(args.output_root, clean_plan, defense, rounds=args.rounds)
        clean_elapsed = 0.0
        clean_reuse_path = reuse_summaries.get((defense, clean_plan.name))
        if clean_reuse_path is not None:
            clean_payload = _load_payload_from_path(clean_reuse_path, round_limit=args.rounds)
        else:
            if args.force or not _summary_complete(clean_dir, rounds=args.rounds):
                clean_elapsed = _run_command(
                    build_attack_command(
                        clean_plan,
                        defense=defense,
                        config_path=args.config,
                        output_root=args.output_root,
                        tb_root=args.tb_root,
                        rounds=args.rounds,
                        num_clients=args.num_clients,
                        num_attackers=args.num_attackers,
                        subsample_rate=args.subsample_rate,
                        krum_attackers=args.krum_attackers,
                        distribution_steps=args.distribution_steps,
                        attack_start_round=args.attack_start_round,
                        policy_train_end_round=args.policy_train_end_round,
                        policy_train_steps_per_round=args.policy_train_steps_per_round,
                        simulator_horizon=args.simulator_horizon,
                        checkpoint_interval=args.checkpoint_interval,
                        parallel_clients=args.parallel_clients,
                        max_client_samples_per_client=args.max_client_samples_per_client,
                        max_eval_samples=args.max_eval_samples,
                    ),
                    log_path=args.output_root / "logs" / f"{defense}_clean.log",
                )
            clean_payload = _load_payload_from_path(clean_dir, round_limit=args.rounds)

        for plan in plans:
            run_dir = expected_run_dir(args.output_root, plan, defense, rounds=args.rounds)
            elapsed = clean_elapsed if plan.name == "clean" else 0.0
            reuse_path = reuse_summaries.get((defense, plan.name))
            if reuse_path is not None:
                attack_payload = _load_payload_from_path(reuse_path, round_limit=args.rounds)
            else:
                if plan.name != "clean" and (args.force or not _summary_complete(run_dir, rounds=args.rounds)):
                    elapsed = _run_command(
                        build_attack_command(
                            plan,
                            defense=defense,
                            config_path=args.config,
                            output_root=args.output_root,
                            tb_root=args.tb_root,
                            rounds=args.rounds,
                            num_clients=args.num_clients,
                            num_attackers=args.num_attackers,
                            subsample_rate=args.subsample_rate,
                            krum_attackers=args.krum_attackers,
                            distribution_steps=args.distribution_steps,
                            attack_start_round=args.attack_start_round,
                            policy_train_end_round=args.policy_train_end_round,
                            policy_train_steps_per_round=args.policy_train_steps_per_round,
                            simulator_horizon=args.simulator_horizon,
                            checkpoint_interval=args.checkpoint_interval,
                            parallel_clients=args.parallel_clients,
                            max_client_samples_per_client=args.max_client_samples_per_client,
                            max_eval_samples=args.max_eval_samples,
                        ),
                        log_path=args.output_root / "logs" / f"{defense}_{plan.name}.log",
                    )
                attack_payload = _load_payload_from_path(run_dir, round_limit=args.rounds)
            rows.append(
                completed_row(
                    defense=defense,
                    attack_name=plan.name,
                    benchmark=plan.benchmark,
                    clean_payload=clean_payload,
                    attack_payload=attack_payload,
                    tail_rounds=args.tail_rounds,
                    elapsed_s=elapsed,
                )
            )
            _write_trend_scalars(
                trend_writer,
                defense=defense,
                attack_name=plan.name,
                clean_payload=clean_payload,
                attack_payload=attack_payload,
            )
            _write_summary_scalars(summary_writer, rows[-1], step=len(rows) - 1)
            _write_matrix_summary_csv(rows, csv_path=args.output_root / "attack_matrix_summary.csv")
    trend_writer.flush()
    trend_writer.close()
    summary_writer.flush()
    summary_writer.close()
    print(f"Matrix summary CSV: {args.output_root / 'attack_matrix_summary.csv'}")
    print(f"Matrix trend TensorBoard: {args.tb_root / 'matrix_trends'}")
    print(f"Matrix summary TensorBoard: {args.tb_root / 'matrix_summary'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--defenses", nargs="+", default=list(DEFAULT_DEFENSES), choices=list(DEFAULT_DEFENSES))
    parser.add_argument("--attacks", nargs="+", default=list(DEFAULT_ATTACK_NAMES), choices=[plan.name for plan in ALL_ATTACK_PLANS])
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--num-clients", dest="num_clients", type=int, default=100)
    parser.add_argument("--num-attackers", dest="num_attackers", type=int, default=20)
    parser.add_argument("--subsample-rate", dest="subsample_rate", type=float, default=0.1)
    parser.add_argument("--krum-attackers", dest="krum_attackers", type=int, default=20)
    parser.add_argument("--distribution-steps", dest="distribution_steps", type=int, default=100)
    parser.add_argument("--attack-start-round", dest="attack_start_round", type=int, default=101)
    parser.add_argument("--policy-train-end-round", dest="policy_train_end_round", type=int, default=150)
    parser.add_argument("--policy-train-steps-per-round", dest="policy_train_steps_per_round", type=int, default=50)
    parser.add_argument("--simulator-horizon", dest="simulator_horizon", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", dest="checkpoint_interval", type=int, default=25)
    parser.add_argument("--parallel-clients", dest="parallel_clients", type=int, default=1)
    parser.add_argument("--max-client-samples-per-client", dest="max_client_samples_per_client", type=int, default=None)
    parser.add_argument("--max-eval-samples", dest="max_eval_samples", type=int, default=None)
    parser.add_argument("--tail-rounds", dest="tail_rounds", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("fl_sandbox/outputs/robust_defense_attack_matrix"))
    parser.add_argument("--tb-root", type=Path, default=Path("fl_sandbox/runs/robust_defense_attack_matrix"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-summary",
        action="append",
        default=[],
        help="Reuse an existing run as defense:attack_name:/path/to/run_dir_or_summary.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_matrix(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
