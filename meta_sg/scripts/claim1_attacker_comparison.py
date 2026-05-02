"""
Claim 1 experiment: RL adaptive attacker vs fixed attacks under the same defense.

Demonstrates that the RL attacker (paper §III-B) achieves higher attack
effectiveness than fixed attacks against an identical defense configuration.

Attack types tested:
  - clean     : no attack (baseline)
  - ipm       : Inner Product Manipulation (untargeted, fixed)
  - lmp       : Local Model Poisoning (untargeted, fixed)
  - bfl       : Backdoor FL (targeted, fixed)
  - dba       : Distributed Backdoor Attack (targeted, fixed)
  - rl        : TD3-based RL attacker (untargeted, adaptive)
  - rl2       : TD3-based RL attacker with stealth-aware robust-defense actions
  - brl       : externally-actioned adaptive backdoor attack
  - sgbrl     : self-guided RL backdoor attack

Metrics:
  - clean_acc  : global test accuracy (lower = more effective for untargeted)
  - backdoor_acc: ASR on backdoor eval set (higher = more effective for targeted)

Usage:
    python meta_sg/scripts/claim1_attacker_comparison.py
    python meta_sg/scripts/claim1_attacker_comparison.py --rounds 80 --device cuda
    python meta_sg/scripts/claim1_attacker_comparison.py --defense krum
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fl_sandbox.attacks.adaptive import RLAttack, RLAttackV2
from fl_sandbox.attacks.adaptive.td3_attacker import RLAttackerConfig
from fl_sandbox.attacks.adaptive.td3_attacker_v2 import RLAttackerConfigV2
from fl_sandbox.attacks.backdoor import BFLAttack, BRLAttack, DBAAttack, SelfGuidedBRLAttack
from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.attacks.vector import IPMAttack, LMPAttack
from fl_sandbox.federation.runner import MinimalFLRunner, SandboxConfig


# ── Condition results ─────────────────────────────────────────────────────────

@dataclass
class ConditionResult:
    attack: str
    defense: str
    clean_accs: List[float] = field(default_factory=list)
    backdoor_accs: List[float] = field(default_factory=list)

    @property
    def final_clean_acc(self) -> float:
        finite = [v for v in self.clean_accs if np.isfinite(v)]
        return finite[-1] if finite else float("nan")

    @property
    def min_clean_acc(self) -> float:
        finite = [v for v in self.clean_accs if np.isfinite(v)]
        return min(finite) if finite else float("nan")

    @property
    def final_backdoor_acc(self) -> float:
        finite = [v for v in self.backdoor_accs if np.isfinite(v)]
        return finite[-1] if finite else float("nan")

    @property
    def max_backdoor_acc(self) -> float:
        finite = [v for v in self.backdoor_accs if np.isfinite(v)]
        return max(finite) if finite else float("nan")


# ── Config builders ───────────────────────────────────────────────────────────

def make_sandbox_config(
    defense: str,
    rounds: int,
    device: str = "cpu",
    seed: int = 42,
) -> SandboxConfig:
    rl_start = max(1, min(10, rounds // 5))
    rl_end   = max(rl_start + 5, min(rounds - 1, 40))
    return SandboxConfig(
        dataset="mnist",
        data_dir="data",
        device=device,
        seed=seed,
        num_clients=10,
        num_attackers=2,
        subsample_rate=0.5,
        local_epochs=1,
        lr=0.05,
        batch_size=32,
        eval_batch_size=512,
        max_client_samples_per_client=250,
        max_eval_samples=2000,
        fltrust_root_size=0,
        defense_type=defense,
        trimmed_mean_ratio=0.2,
        krum_attackers=2,
        clipped_median_norm=2.0,
        rl_attack_start_round=rl_start,
        rl_policy_train_end_round=rl_end,
        rl_distribution_steps=min(10, rounds // 6),
        rl_inversion_steps=30,
        rl_reconstruction_batch_size=4,
        rl_simulator_horizon=6,
        rl_policy_train_episodes_per_round=1,
    )


def build_attacks(config: SandboxConfig) -> Dict[str, Optional[SandboxAttack]]:
    rl_cfg = RLAttackerConfig(
        attack_start_round=config.rl_attack_start_round,
        policy_train_end_round=config.rl_policy_train_end_round,
        distribution_steps=config.rl_distribution_steps,
        inversion_steps=config.rl_inversion_steps,
        reconstruction_batch_size=config.rl_reconstruction_batch_size,
        simulator_horizon=config.rl_simulator_horizon,
        episodes_per_observation=config.rl_policy_train_episodes_per_round,
    )
    rl2_cfg = RLAttackerConfigV2(
        attack_start_round=config.rl_attack_start_round,
        policy_train_end_round=config.rl_policy_train_end_round,
        distribution_steps=config.rl_distribution_steps,
        inversion_steps=config.rl_inversion_steps,
        reconstruction_batch_size=config.rl_reconstruction_batch_size,
        simulator_horizon=config.rl_simulator_horizon,
        episodes_per_observation=max(2, config.rl_policy_train_episodes_per_round),
    )
    return {
        "clean": None,
        "ipm":   IPMAttack(scale=2.0),
        "lmp":   LMPAttack(scale=5.0),
        "bfl":   BFLAttack(poison_frac=1.0),
        "dba":   DBAAttack(num_sub_triggers=4, poison_frac=0.5),
        "rl":    RLAttack(config=rl_cfg),
        "rl2":   RLAttackV2(config=rl2_cfg),
        "brl":   BRLAttack(),
        "sgbrl": SelfGuidedBRLAttack(),
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_condition(
    attack_name: str,
    attack: Optional[SandboxAttack],
    config: SandboxConfig,
    rounds: int,
    verbose: bool = True,
) -> ConditionResult:
    result = ConditionResult(attack=attack_name, defense=config.defense_type)
    runner = MinimalFLRunner(config)
    if verbose:
        print(f"\n  [{attack_name} vs {config.defense_type}] running {rounds} rounds ...", flush=True)
    summaries = runner.run_many_rounds(
        rounds=rounds,
        attack=attack,
        show_progress=verbose,
        progress_desc=f"{attack_name}/{config.defense_type}",
        eval_every=1,
    )
    for s in summaries:
        result.clean_accs.append(float(s.clean_acc))
        result.backdoor_accs.append(float(s.backdoor_acc))
    return result


# ── TensorBoard writer ────────────────────────────────────────────────────────

def log_condition(writer, result: ConditionResult, baseline_clean: Optional[List[float]]) -> None:
    """Log per-round metrics grouped under attack name."""
    for i, (cacc, bacc) in enumerate(zip(result.clean_accs, result.backdoor_accs)):
        t = i + 1
        writer.add_scalar(f"claim1/{result.attack}/clean_acc",    cacc, t)
        writer.add_scalar(f"claim1/{result.attack}/backdoor_acc", bacc, t)
        if baseline_clean is not None and i < len(baseline_clean):
            drop = baseline_clean[i] - cacc
            writer.add_scalar(f"claim1/{result.attack}/clean_acc_drop", drop, t)

    writer.add_scalar(f"summary/final_clean_acc",    result.final_clean_acc,    0)
    writer.add_scalar(f"summary/min_clean_acc",      result.min_clean_acc,      0)
    writer.add_scalar(f"summary/final_backdoor_acc", result.final_backdoor_acc, 0)
    writer.add_scalar(f"summary/max_backdoor_acc",   result.max_backdoor_acc,   0)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: Dict[str, ConditionResult], defense: str) -> None:
    W = 72
    print("\n" + "=" * W)
    print(f"  CLAIM 1: Attack effectiveness under '{defense}' defense")
    print("=" * W)
    hdr = f"{'Attack':<10}  {'Final Clean':>12}  {'Min Clean':>10}  {'Final ASR':>10}  {'Max ASR':>10}"
    print(hdr)
    print("-" * W)

    clean_result = results.get("clean")
    baseline_final = clean_result.final_clean_acc if clean_result else float("nan")

    for name, r in results.items():
        marker = ""
        if name in {"rl", "rl2"}:
            marker = " *"
        elif name == "clean":
            marker = " (baseline)"
        print(
            f"{name + marker:<10}  "
            f"{r.final_clean_acc:>12.4f}  "
            f"{r.min_clean_acc:>10.4f}  "
            f"{r.final_backdoor_acc:>10.4f}  "
            f"{r.max_backdoor_acc:>10.4f}"
        )

    print("=" * W)
    print("  * RL/RL2 are adaptive attackers. Lower clean_acc = more effective untargeted attack.")
    print("    Higher backdoor_acc = more effective targeted attack.")

    if not np.isnan(baseline_final):
        untargeted_fixed = ["ipm", "lmp"]
        adaptive_untargeted = ["rl", "rl2"]
        fixed_drops = {
            name: baseline_final - results[name].final_clean_acc
            for name in untargeted_fixed
            if name in results and np.isfinite(results[name].final_clean_acc)
        }
        adaptive_drops = {
            name: baseline_final - results[name].final_clean_acc
            for name in adaptive_untargeted
            if name in results and np.isfinite(results[name].final_clean_acc)
        }
        if adaptive_drops:
            print("\n  Untargeted clean_acc drops vs baseline:")
            for name, drop in adaptive_drops.items():
                print(f"    {name:<4}: {drop:+.4f}")
            if fixed_drops:
                best_fixed_name, best_fixed_drop = max(fixed_drops.items(), key=lambda item: item[1])
                best_adaptive_name, best_adaptive_drop = max(adaptive_drops.items(), key=lambda item: item[1])
                print(f"  Best fixed attack drop:        {best_fixed_name} {best_fixed_drop:+.4f}")
                if best_adaptive_drop > best_fixed_drop:
                    print(f"  => {best_adaptive_name} is MORE effective than best fixed untargeted attack. Claim 1 SUPPORTED.")
                else:
                    print("  => Adaptive drop not larger than fixed untargeted attacks in this run.")
                    print("     (Try more rounds, RL2, or a statistics-based defense such as krum/trimmed_mean.)")

    targeted_fixed = ["bfl", "dba"]
    adaptive_targeted = ["brl", "sgbrl"]
    targeted_results = {
        name: results[name].final_backdoor_acc
        for name in targeted_fixed + adaptive_targeted
        if name in results and np.isfinite(results[name].final_backdoor_acc)
    }
    if targeted_results:
        print("\n  Targeted ASR comparison:")
        for name, asr in targeted_results.items():
            print(f"    {name:<5}: {asr:.4f}")
        fixed_asrs = {name: targeted_results[name] for name in targeted_fixed if name in targeted_results}
        adaptive_asrs = {name: targeted_results[name] for name in adaptive_targeted if name in targeted_results}
        if fixed_asrs and adaptive_asrs:
            best_fixed_name, best_fixed_asr = max(fixed_asrs.items(), key=lambda item: item[1])
            best_adaptive_name, best_adaptive_asr = max(adaptive_asrs.items(), key=lambda item: item[1])
            if best_adaptive_asr > best_fixed_asr:
                print(f"  => {best_adaptive_name} beats best fixed targeted attack ({best_fixed_name}).")
            else:
                print("  => Adaptive targeted attack does not beat fixed targeted attacks in this run.")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(results: Dict[str, ConditionResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "defense", "round", "clean_acc", "backdoor_acc"])
        for name, r in results.items():
            for i, (cacc, bacc) in enumerate(zip(r.clean_accs, r.backdoor_accs)):
                w.writerow([name, r.defense, i + 1, f"{cacc:.6f}", f"{bacc:.6f}"])
    print(f"  CSV saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds",  type=int, default=60,
                        help="FL rounds per condition (default: 60; RL needs ≥40)")
    parser.add_argument("--device",  default="cpu",  help="torch device (cpu / cuda)")
    parser.add_argument("--defense", default="trimmed_mean",
                        choices=["fedavg", "trimmed_mean", "krum", "clipped_median"],
                        help="fixed defense applied to all conditions")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", default="runs/claim1")
    parser.add_argument("--attacks", default="clean,ipm,lmp,rl2",
                        help="comma-separated list of attack types to run")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_ts   = time.strftime("%Y%m%d-%H%M%S")
    run_tag  = f"{args.defense}_{args.rounds}r_{run_ts}"
    log_dir  = os.path.join(args.out_dir, run_tag)
    csv_path = os.path.join(args.out_dir, f"claim1_{run_tag}.csv")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard log dir: {log_dir}")
    except Exception as exc:
        print(f"[warn] TensorBoard unavailable ({exc}); skipping TB logs.")

    config = make_sandbox_config(
        defense=args.defense,
        rounds=args.rounds,
        device=args.device,
        seed=args.seed,
    )
    print(f"\n[Claim1] defense={args.defense}, rounds={args.rounds}, seed={args.seed}")
    print(f"  RL attack_start={config.rl_attack_start_round}, "
          f"policy_train_end={config.rl_policy_train_end_round}")
    attacks_all = build_attacks(config)

    requested = [a.strip() for a in args.attacks.split(",") if a.strip()]
    attacks = {k: v for k, v in attacks_all.items() if k in requested}

    results: Dict[str, ConditionResult] = {}
    t_total = time.time()
    for attack_name, attack in attacks.items():
        t0 = time.time()
        result = run_condition(attack_name, attack, config, rounds=args.rounds)
        elapsed = time.time() - t0
        results[attack_name] = result
        print(
            f"  [{attack_name}] done in {elapsed:.1f}s — "
            f"final clean={result.final_clean_acc:.4f}, "
            f"final ASR={result.final_backdoor_acc:.4f}"
        )
        if writer is not None:
            baseline = results["clean"].clean_accs if "clean" in results else None
            log_condition(writer, result, baseline)
            writer.flush()

    print(f"\n  Total time: {time.time() - t_total:.1f}s")
    print_summary(results, args.defense)
    save_csv(results, csv_path)

    if writer is not None:
        writer.close()
        print(f"  TensorBoard: tensorboard --logdir {args.out_dir}")


if __name__ == "__main__":
    main()
