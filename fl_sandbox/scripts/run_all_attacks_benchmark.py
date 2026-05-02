"""Benchmark runner: all attack types against FedAvg.

Runs every supported attack type and writes per-run TensorBoard logs under
fl_sandbox/runs/<suite>/<run_name>.

Usage:
    # default (scale=5, 30 rounds)
    python fl_sandbox/scripts/run_all_attacks_benchmark.py

    # scale=2, 100 rounds — more detailed / longer run
    python fl_sandbox/scripts/run_all_attacks_benchmark.py --scale 2 --rounds 100

Open TensorBoard:
    bash fl_sandbox/scripts/run_tensorboard.sh fl_sandbox/runs/all_attacks_scale2_100r
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.config import RunConfig, config_to_namespace
from fl_sandbox.core.attacks import ATTACK_CHOICES
from fl_sandbox.core.experiment_builders import build_run_name
from fl_sandbox.core.experiment_service import (
    execute_experiment,
    persist_experiment_artifacts,
)

# Common FL settings shared by every run
COMMON = dict(
    num_clients=20,
    num_attackers=4,
    subsample_rate=0.5,
    local_epochs=1,
    lr=0.05,
    batch_size=64,
    eval_batch_size=1024,
    num_workers=0,
    seed=42,
    dataset="mnist",
    split_mode="iid",
    krum_attackers=1,
    device="auto",
)

# ASR threshold for "converged"
_CONVERGE_THRESHOLD = 0.9
# Number of tail rounds used for stable-mean metrics
_TAIL_ROUNDS = 10


def _attack_overrides(scale: float, rounds: int) -> dict[str, dict]:
    # RL warmup proportional to rounds; more inversion steps for longer runs
    warmup = max(10, rounds // 5)
    inversion_steps = min(100, max(20, rounds // 2))
    return {
        "clean":     {},
        "ipm":       dict(ipm_scaling=scale),
        "lmp":       dict(lmp_scale=scale),
        "alie":      {},   # tau uses default 1.5; no scale dependency
        "signflip":  {},   # no parameters
        "gaussian":  {},   # sigma uses default 0.01
        "bfl":       dict(bfl_poison_frac=1.0),
        "dba":   dict(dba_poison_frac=0.5, dba_num_sub_triggers=4),
        "brl":   dict(attacker_action=[0.0, 0.0, 0.0]),
        "sgbrl": {},
        "rl": dict(
            protocol="rlfl",
            warmup_rounds=warmup,
            # 0 → normalize() derives from warmup_rounds
            rl_distribution_steps=0,
            rl_attack_start_round=0,
            # Train policy throughout all rounds (not just warmup)
            rl_policy_train_end_round=rounds,
            rl_inversion_steps=inversion_steps,
            rl_reconstruction_batch_size=8,
            rl_policy_train_episodes_per_round=2,
            rl_simulator_horizon=10,
        ),
        "rl2": dict(
            protocol="rlfl",
            warmup_rounds=warmup,
            rl_distribution_steps=0,
            rl_attack_start_round=0,
            rl_policy_train_end_round=rounds,
            rl_inversion_steps=inversion_steps,
            rl_reconstruction_batch_size=8,
            rl_policy_train_episodes_per_round=2,
            rl_simulator_horizon=10,
        ),
    }


def _make_args(
    attack_type: str,
    scale: float,
    rounds: int,
    output_root: str,
    tb_root: str,
    defense_type: str = "fedavg",
    device: Optional[str] = None,
    parallel_clients: Optional[int] = None,
    max_client_samples_per_client: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    rl_episodes: Optional[int] = None,
    rl_horizon: Optional[int] = None,
    rl_inversion_steps: Optional[int] = None,
) -> argparse.Namespace:
    cfg = dict(COMMON)
    cfg["rounds"] = rounds
    cfg["defense_type"] = defense_type
    cfg.update(_attack_overrides(scale, rounds).get(attack_type, {}))
    if device is not None:
        cfg["device"] = device
    if parallel_clients is not None:
        cfg["parallel_clients"] = parallel_clients
    if max_client_samples_per_client is not None:
        cfg["max_client_samples_per_client"] = max_client_samples_per_client
    if max_eval_samples is not None:
        cfg["max_eval_samples"] = max_eval_samples
    if attack_type in {"rl", "rl2"}:
        if rl_episodes is not None:
            cfg["rl_policy_train_episodes_per_round"] = rl_episodes
        if rl_horizon is not None:
            cfg["rl_simulator_horizon"] = rl_horizon
        if rl_inversion_steps is not None:
            cfg["rl_inversion_steps"] = rl_inversion_steps
    cfg["attack_type"] = attack_type
    cfg.setdefault("protocol", "none")
    cfg.setdefault("warmup_rounds", 0)
    cfg["output_root"] = output_root
    cfg["tb_root"] = tb_root
    cfg["config"] = ""
    cfg["output_dir"] = ""
    cfg["tb_dir"] = ""
    if "attacker_action" in cfg:
        cfg["attacker_action"] = tuple(cfg["attacker_action"])
    return argparse.Namespace(**cfg)


def _prepare(args: argparse.Namespace):
    run_config = RunConfig.from_flat_dict(
        {k: v for k, v in vars(args).items() if k not in {"config", "output_dir", "tb_dir"}}
    )
    run_name = build_run_name(
        dataset=run_config.data.dataset,
        attack_type=run_config.attacker.type,
        defense_type=run_config.defender.type,
        split_mode=run_config.data.split_mode,
        noniid_q=run_config.data.noniid_q,
        rounds=run_config.runtime.rounds,
    )
    run_args = config_to_namespace(run_config)
    run_args.output_dir = str(Path(args.output_root) / run_name)
    run_args.tb_dir = str(Path(args.tb_root) / run_name)
    for field in ("output_root", "tb_root", "config"):
        setattr(run_args, field, getattr(args, field))
    return run_args, run_name, run_config


def _detailed_metrics(series: dict, rounds: int) -> dict:
    """Compute richer per-run metrics from the full round-by-round series."""
    asr = [v for v in series.get("asr", []) if not math.isnan(v)]
    clean = [v for v in series.get("clean_acc", []) if not math.isnan(v)]
    mal_norms = [v for v in series.get("mean_malicious_norm", []) if not math.isnan(v)]
    benign_norms = [v for v in series.get("mean_benign_norm", []) if not math.isnan(v)]

    tail = _TAIL_ROUNDS
    peak_asr = float(max(asr)) if asr else float("nan")
    last_mean_asr = float(np.mean(asr[-tail:])) if len(asr) >= tail else float(np.mean(asr)) if asr else float("nan")
    last_mean_clean = float(np.mean(clean[-tail:])) if len(clean) >= tail else float(np.mean(clean)) if clean else float("nan")
    mean_mal_norm = float(np.nanmean(mal_norms)) if mal_norms else float("nan")
    mean_benign_norm = float(np.nanmean(benign_norms)) if benign_norms else float("nan")

    # First round ASR crosses the convergence threshold
    convergence_round = next(
        (i + 1 for i, v in enumerate(series.get("asr", [])) if not math.isnan(v) and v >= _CONVERGE_THRESHOLD),
        None,
    )

    return {
        "peak_asr": round(peak_asr, 4),
        "last10_mean_asr": round(last_mean_asr, 4),
        "last10_mean_clean_acc": round(last_mean_clean, 4),
        "convergence_round": convergence_round,
        "mean_mal_norm": round(mean_mal_norm, 4),
        "mean_benign_norm": round(mean_benign_norm, 4),
    }


def run_all(
    attacks: Optional[list[str]] = None,
    scale: float = 5.0,
    rounds: int = 30,
    defense_type: str = "fedavg",
    device: Optional[str] = None,
    parallel_clients: Optional[int] = None,
    max_client_samples_per_client: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    rl_episodes: Optional[int] = None,
    rl_horizon: Optional[int] = None,
    rl_inversion_steps: Optional[int] = None,
) -> None:
    attacks = attacks or list(ATTACK_CHOICES)
    suite = f"{defense_type}_scale{scale:g}_{rounds}r"
    output_root = f"fl_sandbox/outputs/all_attacks_{suite}"
    tb_root = f"fl_sandbox/runs/all_attacks_{suite}"

    total = len(attacks)
    summary_rows: list[dict] = []
    overall_start = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"All-attacks benchmark  —  scale={scale:g}  rounds={rounds}  ({total} runs)")
    print(f"Clients: {COMMON['num_clients']}  Attackers: {COMMON['num_attackers']}  "
          f"SubsampleRate: {COMMON['subsample_rate']}  Defense: {defense_type}")
    print(f"TensorBoard root: {tb_root}")
    print(f"{'='*70}\n")

    for idx, attack_type in enumerate(attacks, 1):
        args = _make_args(
            attack_type,
            scale,
            rounds,
            output_root,
            tb_root,
            defense_type,
            device=device,
            parallel_clients=parallel_clients,
            max_client_samples_per_client=max_client_samples_per_client,
            max_eval_samples=max_eval_samples,
            rl_episodes=rl_episodes,
            rl_horizon=rl_horizon,
            rl_inversion_steps=rl_inversion_steps,
        )
        run_args, run_name, run_config = _prepare(args)

        print(f"[{idx}/{total}] {run_name}")
        t0 = time.perf_counter()
        try:
            result = execute_experiment(run_args, progress_desc=run_name, run_config=run_config)
            persist_experiment_artifacts(
                result,
                write_client_metrics=True,
                write_tensorboard=True,
                write_round_metrics=True,
            )
            elapsed = time.perf_counter() - t0
            final = result.payload["final"]
            series = result.payload.get("series", {})
            detail = _detailed_metrics(series, rounds)
            row = {
                "run_name": run_name,
                "attack_type": attack_type,
                "scale": scale,
                "rounds": rounds,
                "defense_type": defense_type,
                # Final-round metrics
                "final_clean_acc": round(float(final["clean_acc"]), 4),
                "final_asr": round(float(final["asr"]), 4),
                "final_mal_norm": round(float(final["mean_malicious_norm"]), 4),
                # Detailed metrics
                **detail,
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)
            conv = f"r{detail['convergence_round']}" if detail["convergence_round"] else "never"
            print(
                f"  ✓ clean={row['final_clean_acc']:.4f}  "
                f"asr={row['final_asr']:.4f}  "
                f"peak_asr={detail['peak_asr']:.4f}  "
                f"converge={conv}  "
                f"l10asr={detail['last10_mean_asr']:.4f}  "
                f"({elapsed:.1f}s)\n"
                f"    TB: {run_args.tb_dir}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ✗ FAILED after {elapsed:.1f}s: {exc}")
            import traceback; traceback.print_exc()
            summary_rows.append({"run_name": run_name, "attack_type": attack_type, "error": str(exc)})
        print()

    total_elapsed = time.perf_counter() - overall_start
    print(f"{'='*70}")
    print(f"All done in {total_elapsed:.1f}s  (scale={scale:g}, rounds={rounds})\n")

    # Detailed summary table
    w = 9
    header = (
        f"{'Attack':<8}  {'CleanAcc':>{w}}  {'FinalASR':>{w}}  {'PeakASR':>{w}}  "
        f"{'L10ASR':>{w}}  {'Converge':>8}  {'MalNorm':>{w}}  {'Sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        if "error" in row:
            print(f"{row['attack_type']:<8}  FAILED: {row['error']}")
        else:
            conv = f"r{row['convergence_round']}" if row.get("convergence_round") else "never"
            print(
                f"{row['attack_type']:<8}  "
                f"{row['final_clean_acc']:>{w}.4f}  "
                f"{row['final_asr']:>{w}.4f}  "
                f"{row['peak_asr']:>{w}.4f}  "
                f"{row['last10_mean_asr']:>{w}.4f}  "
                f"{conv:>8}  "
                f"{row['final_mal_norm']:>{w}.4f}  "
                f"{row['elapsed_s']:>6.1f}"
            )

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)
    print(f"\nSummary JSON : {summary_path}")
    print(f"TensorBoard  : bash fl_sandbox/scripts/run_tensorboard.sh {tb_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all-attacks benchmark")
    parser.add_argument("--scale", type=float, default=5.0,
                        help="Scaling factor for IPM/LMP attacks (default: 5.0)")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Number of FL rounds per run (default: 30)")
    parser.add_argument("--attacks", nargs="+", choices=list(ATTACK_CHOICES),
                        default=None, help="Subset of attacks to run (default: all)")
    parser.add_argument("--defense", default="fedavg",
                        help="Defense type to use (default: fedavg)")
    parser.add_argument("--device", default=None,
                        help="Torch device override: auto / cpu / mps / cuda")
    parser.add_argument("--parallel-clients", type=int, default=None,
                        help="Train benign clients in parallel where supported")
    parser.add_argument("--max-client-samples", type=int, default=None,
                        help="Cap local training samples per client for faster smoke runs")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Cap evaluation samples for faster smoke runs")
    parser.add_argument("--rl-episodes", type=int, default=None,
                        help="Override RL policy-training simulator episodes per FL round")
    parser.add_argument("--rl-horizon", type=int, default=None,
                        help="Override RL simulator horizon")
    parser.add_argument("--rl-inversion-steps", type=int, default=None,
                        help="Override RL gradient inversion steps")
    cli = parser.parse_args()
    run_all(
        attacks=cli.attacks,
        scale=cli.scale,
        rounds=cli.rounds,
        defense_type=cli.defense,
        device=cli.device,
        parallel_clients=cli.parallel_clients,
        max_client_samples_per_client=cli.max_client_samples,
        max_eval_samples=cli.max_eval_samples,
        rl_episodes=cli.rl_episodes,
        rl_horizon=cli.rl_horizon,
        rl_inversion_steps=cli.rl_inversion_steps,
    )
