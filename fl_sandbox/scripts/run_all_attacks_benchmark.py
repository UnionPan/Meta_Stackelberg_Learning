"""Benchmark runner: all attack types, 30 rounds, 20 clients, 4 attackers.

Runs every supported attack type against FedAvg and writes per-run TensorBoard
logs under fl_sandbox/runs/<suite>/<run_name>.

Usage:
    # default suite (ipm_scaling=5, lmp_scale=5)
    python fl_sandbox/scripts/run_all_attacks_benchmark.py

    # scale=2 suite
    python fl_sandbox/scripts/run_all_attacks_benchmark.py --scale 2

Open TensorBoard:
    bash fl_sandbox/scripts/run_tensorboard.sh fl_sandbox/runs/all_attacks_scale5_30r
    bash fl_sandbox/scripts/run_tensorboard.sh fl_sandbox/runs/all_attacks_scale2_30r
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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
    rounds=30,
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
    defense_type="fedavg",
    krum_attackers=1,
    device="auto",
)


def _attack_overrides(scale: float) -> dict[str, dict]:
    return {
        "clean": {},
        "ipm":   dict(ipm_scaling=scale),
        "lmp":   dict(lmp_scale=scale),
        "bfl":   dict(bfl_poison_frac=1.0),
        "dba":   dict(dba_poison_frac=0.5, dba_num_sub_triggers=4),
        "brl":   dict(attacker_action=[0.0, 0.0, 0.0]),
        "rl": dict(
            protocol="rlfl",
            warmup_rounds=5,
            rl_inversion_steps=20,
            rl_reconstruction_batch_size=8,
            rl_policy_train_episodes_per_round=1,
            rl_simulator_horizon=5,
        ),
    }


def _make_args(attack_type: str, scale: float, output_root: str, tb_root: str) -> argparse.Namespace:
    cfg = dict(COMMON)
    cfg.update(_attack_overrides(scale).get(attack_type, {}))
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


def run_all(attacks: Optional[list[str]] = None, scale: float = 5.0) -> None:
    attacks = attacks or list(ATTACK_CHOICES)
    suite = f"scale{scale:g}"
    output_root = f"fl_sandbox/outputs/all_attacks_{suite}_30r"
    tb_root = f"fl_sandbox/runs/all_attacks_{suite}_30r"

    total = len(attacks)
    summary_rows: list[dict] = []
    overall_start = time.perf_counter()

    print(f"\n{'='*62}")
    print(f"All-attacks benchmark  —  scale={scale:g}  ({total} runs)")
    print(f"Clients: {COMMON['num_clients']}  Attackers: {COMMON['num_attackers']}  "
          f"Rounds: {COMMON['rounds']}  Defense: {COMMON['defense_type']}")
    print(f"TensorBoard root: {tb_root}")
    print(f"{'='*62}\n")

    for idx, attack_type in enumerate(attacks, 1):
        args = _make_args(attack_type, scale, output_root, tb_root)
        run_args, run_name, run_config = _prepare(args)

        print(f"[{idx}/{total}] Starting: {run_name}")
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
            row = {
                "run_name": run_name,
                "attack_type": attack_type,
                "scale": scale,
                "defense_type": COMMON["defense_type"],
                "final_clean_acc": round(float(final["clean_acc"]), 4),
                "final_backdoor_acc": round(float(final["backdoor_acc"]), 4),
                "final_asr": round(float(final["asr"]), 4),
                "mean_malicious_norm": round(float(final["mean_malicious_norm"]), 4),
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)
            print(
                f"  ✓ clean_acc={row['final_clean_acc']:.4f}  "
                f"asr={row['final_asr']:.4f}  "
                f"mal_norm={row['mean_malicious_norm']:.4f}  "
                f"({elapsed:.1f}s)\n"
                f"    TB: {run_args.tb_dir}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ✗ FAILED after {elapsed:.1f}s: {exc}")
            summary_rows.append({"run_name": run_name, "attack_type": attack_type, "error": str(exc)})
        print()

    total_elapsed = time.perf_counter() - overall_start
    print(f"{'='*62}")
    print(f"All done in {total_elapsed:.1f}s  (scale={scale:g})\n")
    print(f"{'Attack':<8}  {'CleanAcc':>8}  {'ASR':>6}  {'MalNorm':>9}  {'Sec':>6}")
    print("-" * 52)
    for row in summary_rows:
        if "error" in row:
            print(f"{row['attack_type']:<8}  FAILED: {row['error']}")
        else:
            print(
                f"{row['attack_type']:<8}  {row['final_clean_acc']:>8.4f}  "
                f"{row['final_asr']:>6.4f}  {row['mean_malicious_norm']:>9.4f}  "
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
    parser = argparse.ArgumentParser(description="Run all attack types benchmark")
    parser.add_argument("--scale", type=float, default=5.0,
                        help="Scaling factor for IPM/LMP attacks (default: 5.0)")
    parser.add_argument("--attacks", nargs="+", choices=list(ATTACK_CHOICES),
                        default=None, help="Subset of attacks (default: all)")
    cli = parser.parse_args()
    run_all(attacks=cli.attacks, scale=cli.scale)
