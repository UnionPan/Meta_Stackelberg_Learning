"""
Claim 1 Validation: RL adaptive attacker vs fixed attacks under a real defense.

Experimental design:
  Defense : clipped_median (norm_bound=2.0)  — blocks naive large-norm attacks
  Rounds  : 80  — RL needs ~30 rounds to warm up; final 50 show steady-state
  Scale   : 2   — fair fixed-attack strength (not hand-tuned to win)
  Attacks : clean, ipm, lmp, bfl, dba, rl

Expected outcome (Claim 1):
  Fixed attacks (IPM/LMP) have their large updates clipped → low clean_acc damage
  Backdoor attacks (BFL/DBA) may be partially blocked by the norm bound
  RL attacker learns smaller, stealthy updates that pass through clipping → higher damage

Output:
  TensorBoard : fl_sandbox/runs/claim1_validation/<attack>_clipped_median_iid_80r/
  JSON summary: fl_sandbox/outputs/claim1_validation/benchmark_summary.json

Run:
    python fl_sandbox/scripts/run_claim1_validation.py
    python fl_sandbox/scripts/run_claim1_validation.py --rounds 100 --device cpu
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
from fl_sandbox.core.experiment_builders import build_run_name
from fl_sandbox.core.experiment_service import (
    execute_experiment,
    persist_experiment_artifacts,
)

# ── FL Setup ──────────────────────────────────────────────────────────────────

SUITE_NAME = "claim1_validation"

FL_COMMON = dict(
    num_clients=20,
    num_attackers=4,       # 20% of all clients → sampled ~40% of rounds
    subsample_rate=0.5,
    local_epochs=1,
    lr=0.05,
    batch_size=64,
    eval_batch_size=1024,
    num_workers=0,
    seed=42,
    dataset="mnist",
    split_mode="iid",
    krum_attackers=2,      # krum param: assume up to 2 attackers known
    # Defense-specific
    clipped_median_norm=2.0,     # clip L2 norm of each update to ≤ 2.0
    trimmed_mean_ratio=0.3,      # trim 30% from each tail (more than attacker ratio)
)

SCALE = 2.0        # IPM/LMP scaling — moderate, not hand-tuned extreme
DEFENSE = "clipped_median"
TAIL_ROUNDS = 15   # rounds used for last-N mean metrics
_ASR_CONVERGE = 0.9


# ── Per-attack parameter overrides ────────────────────────────────────────────

def _attack_overrides(rounds: int) -> dict[str, dict]:
    warmup = max(10, rounds // 6)
    inversion_steps = min(80, max(30, rounds // 2))
    return {
        "clean": {},
        "ipm":   dict(ipm_scaling=SCALE),
        "lmp":   dict(lmp_scale=SCALE),
        "bfl":   dict(bfl_poison_frac=1.0),
        "dba":   dict(dba_poison_frac=0.5, dba_num_sub_triggers=4),
        "brl":  dict(attacker_action=[0.0, 0.0, 0.0]),
        "rl": dict(
            protocol="rlfl",
            warmup_rounds=warmup,
            rl_distribution_steps=0,      # derived from warmup by normalize()
            rl_attack_start_round=0,       # derived from warmup by normalize()
            rl_policy_train_end_round=rounds,   # keep training through all rounds
            rl_inversion_steps=inversion_steps,
            rl_reconstruction_batch_size=8,
            rl_policy_train_episodes_per_round=2,
            rl_simulator_horizon=10,
        ),
    }


# ── Experiment builder ────────────────────────────────────────────────────────

def _make_args(
    attack_type: str,
    rounds: int,
    output_root: str,
    tb_root: str,
    device: str,
) -> argparse.Namespace:
    cfg = dict(FL_COMMON)
    cfg["rounds"] = rounds
    cfg["defense_type"] = DEFENSE
    cfg["device"] = device
    cfg.update(_attack_overrides(rounds).get(attack_type, {}))
    cfg["attack_type"] = attack_type
    cfg.setdefault("protocol", "none")
    cfg.setdefault("warmup_rounds", 0)
    cfg["output_root"] = output_root
    cfg["tb_root"] = tb_root
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
    for field in ("output_root", "tb_root"):
        setattr(run_args, field, getattr(args, field))
    return run_args, run_name, run_config


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _tail_metrics(series: dict, rounds: int) -> dict:
    asr   = [v for v in series.get("asr",       []) if not math.isnan(v)]
    clean = [v for v in series.get("clean_acc", []) if not math.isnan(v)]
    mal   = [v for v in series.get("mean_malicious_norm", []) if not math.isnan(v)]
    benign= [v for v in series.get("mean_benign_norm",    []) if not math.isnan(v)]

    n = TAIL_ROUNDS
    peak_asr       = float(max(asr))   if asr   else float("nan")
    last_mean_asr  = float(np.mean(asr[-n:]))   if len(asr)   >= n else float(np.mean(asr))   if asr   else float("nan")
    last_mean_clean= float(np.mean(clean[-n:])) if len(clean) >= n else float(np.mean(clean)) if clean else float("nan")
    mean_mal       = float(np.nanmean(mal))    if mal    else float("nan")
    mean_benign    = float(np.nanmean(benign)) if benign else float("nan")
    converge_round = next(
        (i + 1 for i, v in enumerate(series.get("asr", [])) if not math.isnan(v) and v >= _ASR_CONVERGE),
        None,
    )
    return {
        "peak_asr":           round(peak_asr,        4),
        "last15_mean_asr":    round(last_mean_asr,   4),
        "last15_mean_clean":  round(last_mean_clean, 4),
        "convergence_round":  converge_round,
        "mean_mal_norm":      round(mean_mal,    4),
        "mean_benign_norm":   round(mean_benign, 4),
    }


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_claim1(
    rounds: int = 80,
    attacks: Optional[list[str]] = None,
    device: str = "auto",
) -> None:
    attacks = attacks or ["clean", "ipm", "lmp", "bfl", "dba", "rl", "brl"]
    output_root = f"fl_sandbox/outputs/{SUITE_NAME}"
    tb_root     = f"fl_sandbox/runs/{SUITE_NAME}"

    Path(output_root).mkdir(parents=True, exist_ok=True)
    Path(tb_root).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"Claim 1 Validation  —  defense={DEFENSE}  rounds={rounds}")
    print(f"Clients: {FL_COMMON['num_clients']}  Attackers: {FL_COMMON['num_attackers']}  "
          f"Subsample: {FL_COMMON['subsample_rate']}  Scale: {SCALE}")
    print(f"clipped_median_norm={FL_COMMON['clipped_median_norm']}")
    print(f"TensorBoard: tensorboard --logdir {tb_root}")
    print(f"{'='*72}\n")

    summary_rows: list[dict] = []
    t_total = time.perf_counter()

    for idx, attack_type in enumerate(attacks, 1):
        args = _make_args(attack_type, rounds, output_root, tb_root, device)
        run_args, run_name, run_config = _prepare(args)

        print(f"[{idx}/{len(attacks)}] {run_name}")
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
            final  = result.payload["final"]
            series = result.payload.get("series", {})
            detail = _tail_metrics(series, rounds)
            row = {
                "run_name":        run_name,
                "attack_type":     attack_type,
                "defense":         DEFENSE,
                "rounds":          rounds,
                "scale":           SCALE,
                "final_clean_acc": round(float(final["clean_acc"]), 4),
                "final_asr":       round(float(final["asr"]),       4),
                "final_mal_norm":  round(float(final["mean_malicious_norm"]), 4),
                **detail,
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)
            conv = f"r{detail['convergence_round']}" if detail["convergence_round"] else "never"
            print(
                f"  clean={row['final_clean_acc']:.4f}  "
                f"asr={row['final_asr']:.4f}  "
                f"peak_asr={detail['peak_asr']:.4f}  "
                f"l15asr={detail['last15_mean_asr']:.4f}  "
                f"converge={conv}  "
                f"({elapsed:.0f}s)\n"
                f"    TB → {run_args.tb_dir}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED after {elapsed:.0f}s: {exc}")
            import traceback; traceback.print_exc()
            summary_rows.append({"run_name": run_name, "attack_type": attack_type, "error": str(exc)})
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    print(f"{'='*72}")
    print(f"Done in {total_elapsed/60:.1f} min  |  defense={DEFENSE}  rounds={rounds}\n")

    baseline_clean = next(
        (r["last15_mean_clean"] for r in summary_rows if r.get("attack_type") == "clean"),
        float("nan"),
    )

    hdr = f"{'Attack':<8}  {'CleanAcc':>9}  {'FinalASR':>9}  {'PeakASR':>9}  {'L15ASR':>9}  {'MalNorm':>8}  {'Drop':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in summary_rows:
        if "error" in row:
            print(f"{row['attack_type']:<8}  FAILED: {row['error']}")
            continue
        drop = baseline_clean - row["last15_mean_clean"] if not math.isnan(baseline_clean) else float("nan")
        marker = " *" if row["attack_type"] == "rl" else ""
        print(
            f"{row['attack_type'] + marker:<8}  "
            f"{row['final_clean_acc']:>9.4f}  "
            f"{row['final_asr']:>9.4f}  "
            f"{row['peak_asr']:>9.4f}  "
            f"{row['last15_mean_asr']:>9.4f}  "
            f"{row['final_mal_norm']:>8.3f}  "
            f"{drop:>+7.4f}"
        )
    print(f"\n* = adaptive RL attacker   Drop = baseline_clean - last15_mean_clean")

    # Untargeted: RL vs IPM/LMP
    rl_row = next((r for r in summary_rows if r.get("attack_type") == "rl" and "error" not in r), None)
    if rl_row and not math.isnan(baseline_clean):
        fixed_drops = [
            baseline_clean - r["last15_mean_clean"]
            for r in summary_rows
            if r.get("attack_type") in {"ipm", "lmp"} and "error" not in r
        ]
        rl_drop = baseline_clean - rl_row["last15_mean_clean"]
        best_fixed = max(fixed_drops) if fixed_drops else float("nan")
        print(f"\n[Untargeted] RL clean_acc drop (last 15r): {rl_drop:+.4f}")
        if not math.isnan(best_fixed):
            print(f"[Untargeted] Best fixed drop (IPM/LMP):    {best_fixed:+.4f}")
            verdict = "SUPPORTED" if rl_drop > best_fixed else "not yet — may need more rounds"
            print(f"Claim 1 untargeted [{verdict}]")

    # Targeted: BRL vs BFL/DBA
    brl_row = next((r for r in summary_rows if r.get("attack_type") == "brl" and "error" not in r), None)
    if brl_row:
        fixed_asr = [
            r["last15_mean_asr"]
            for r in summary_rows
            if r.get("attack_type") in {"bfl", "dba"} and "error" not in r
        ]
        brl_asr = brl_row["last15_mean_asr"]
        best_fixed_asr = max(fixed_asr) if fixed_asr else float("nan")
        print(f"\n[Targeted]   BRL ASR (last 15r):           {brl_asr:+.4f}")
        if not math.isnan(best_fixed_asr):
            print(f"[Targeted]   Best fixed ASR (BFL/DBA):     {best_fixed_asr:+.4f}")
            verdict = "SUPPORTED" if brl_asr > best_fixed_asr else "not yet"
            print(f"Claim 1 targeted  [{verdict}]")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = Path(output_root) / "benchmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print(f"TensorBoard:\n  tensorboard --logdir {tb_root} --port 6009")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds",  type=int, default=80,
                        help="FL rounds per attack (default: 80)")
    parser.add_argument("--device",  default="auto",
                        help="torch device: auto / cpu / cuda (default: auto)")
    parser.add_argument("--attacks", nargs="+",
                        default=["clean", "ipm", "lmp", "bfl", "dba", "rl", "brl"],
                        help="Attack types to include")
    args = parser.parse_args()
    run_claim1(rounds=args.rounds, attacks=args.attacks, device=args.device)
