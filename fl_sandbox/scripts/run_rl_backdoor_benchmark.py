"""Paired-seed defense-matrix benchmark for the RL backdoor attacker.

Drives ``run_experiment.py`` over the cross product (attack ∈ {rl_backdoor,
bfl, dba, fixed_action_rl_backdoor}) × (defense ∈ {fedavg, krum, multi_krum,
median, clipped_median, trimmed_mean}) × (seed ∈ args.seeds), all sharing the
same per-run base config. Paired seeds (I3 closed): the same seed list runs
every (attack, defense) cell so the comparison is matched, not offset by
``seed + N*k`` ad-hoc shifts.

Each cell is one ``run_experiment.py --config <derived.yaml>`` invocation, so
the runner code path is identical to a single-config experiment — no bespoke
training loop, no separate output schema.

The script collects ``summary.json`` from every cell and emits a wide CSV with
mean ± std of clean accuracy, backdoor ASR, and the sim→real transfer-gap
metric ``rl_sim2real_gap`` across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_DEFENSES = (
    "fedavg",
    "krum",
    "multi_krum",
    "median",
    "clipped_median",
    "trimmed_mean",
)

DEFAULT_ATTACKS = ("rl_backdoor", "bfl", "dba")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("fl_sandbox/config/rl_backdoor.example.yaml"),
        help="YAML config used as the base for every cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fl_sandbox/outputs/rl_backdoor_benchmark"),
    )
    parser.add_argument(
        "--tb-root",
        type=Path,
        default=None,
        help="TensorBoard root. Defaults to a tb directory under each output cell.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[150, 151, 152])
    parser.add_argument("--attacks", type=str, nargs="+", default=list(DEFAULT_ATTACKS))
    parser.add_argument("--defenses", type=str, nargs="+", default=list(DEFAULT_DEFENSES))
    parser.add_argument("--rounds", type=int, default=None, help="Override total rounds.")
    parser.add_argument("--max-client-samples-per-client", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--rl-backdoor-policy-checkpoint",
        type=Path,
        default=None,
        help="Pretrained policy checkpoint; if set, the rl_backdoor cell deploys it frozen.",
    )
    parser.add_argument(
        "--rl-backdoor-separate-frozen-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train rl_backdoor policies in a separate first stage, then benchmark "
            "a fresh frozen-policy FL run. Enabled by default."
        ),
    )
    parser.add_argument(
        "--rl-backdoor-train-rounds",
        type=int,
        default=400,
        help="Training-stage rounds for rl_backdoor policy checkpoints.",
    )
    parser.add_argument(
        "--rl-backdoor-attack-start-round",
        type=int,
        default=101,
        help="First live attack/policy-training round for the rl_backdoor training stage.",
    )
    parser.add_argument(
        "--rl-backdoor-policy-train-end-round",
        type=int,
        default=400,
        help="Last policy-training round for the rl_backdoor training stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run commands without executing them.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _dump_yaml(payload: dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)


def _cell_config(base: dict, *, attack: str, defense: str, seed: int, args) -> dict:
    cfg = json.loads(json.dumps(base))  # deep copy
    cfg.setdefault("runtime", {})["seed"] = int(seed)
    if args.rounds is not None:
        cfg["runtime"]["rounds"] = int(args.rounds)
    if getattr(args, "max_client_samples_per_client", None) is not None:
        cfg["runtime"]["max_client_samples_per_client"] = int(args.max_client_samples_per_client)
    if getattr(args, "max_eval_samples", None) is not None:
        cfg["runtime"]["max_eval_samples"] = int(args.max_eval_samples)
    cfg.setdefault("attacker", {})["type"] = attack
    cfg.setdefault("defender", {})["type"] = defense
    if attack == "rl_backdoor" and args.rl_backdoor_policy_checkpoint:
        cfg["attacker"]["rl_freeze_policy"] = True
        cfg["attacker"]["rl_policy_checkpoint_path"] = str(args.rl_backdoor_policy_checkpoint)
        cfg["attacker"]["rl_policy_checkpoint_dir"] = ""
        cfg["attacker"]["rl_policy_train_end_round"] = 0
        cfg["attacker"]["rl_checkpoint_interval"] = 0
        cfg["attacker"]["rl_save_final_checkpoint"] = False
    elif attack == "rl_backdoor":
        cfg["attacker"]["rl_freeze_policy"] = False
        cfg["attacker"]["rl_policy_checkpoint_path"] = ""
        cfg["attacker"]["rl_policy_checkpoint_dir"] = ""
        cfg["attacker"]["rl_attack_start_round"] = int(args.rl_backdoor_attack_start_round)
        cfg["attacker"]["rl_policy_train_end_round"] = int(args.rl_backdoor_policy_train_end_round)
        cfg["attacker"]["rl_save_final_checkpoint"] = True
    cell_root = args.output_dir / f"{attack}_{defense}_seed{seed}"
    cfg.setdefault("output", {})["output_root"] = str(cell_root)
    tb_root = getattr(args, "tb_root", None)
    cfg["output"]["tb_root"] = (
        str(Path(tb_root) / f"{attack}_{defense}_seed{seed}") if tb_root else str(cell_root / "tb")
    )
    return cfg


def _run_cell(cfg_path: Path, *, dry_run: bool) -> int:
    cmd = [sys.executable, "-m", "fl_sandbox.run.run_experiment", "--config", str(cfg_path)]
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _args_with(args, **overrides):
    payload = vars(args).copy()
    payload.update(overrides)
    return argparse.Namespace(**payload)


def _policy_checkpoint_for_cell(cell_root: Path) -> Path | None:
    candidates = sorted(cell_root.glob("*/checkpoints/rl_policy_latest.pt"))
    return candidates[0] if candidates else None


def _load_summary(cell_root: Path) -> dict | None:
    # ``run_experiment.py`` writes to ``output_root/<run_name>/summary.json``
    # (run_name is the dataset_attack_defense_split_rounds slug built by
    # ``build_run_name``), so glob one level deep rather than reading the
    # cell root directly.
    candidates = sorted(cell_root.glob("*/summary.json"))
    if not candidates:
        direct = cell_root / "summary.json"
        if direct.is_file():
            candidates = [direct]
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _mean_post_training(values: list[float], window: int = 50) -> float | None:
    finite = [v for v in values if v is not None and isinstance(v, (int, float)) and v == v]
    if not finite:
        return None
    tail = finite[-min(len(finite), window):]
    return statistics.fmean(tail) if tail else None


def _aggregate(
    output_dir: Path, attacks: Iterable[str], defenses: Iterable[str], seeds: Iterable[int]
) -> List[dict]:
    rows: List[dict] = []
    for attack in attacks:
        for defense in defenses:
            clean = []
            asr = []
            gaps = []
            for seed in seeds:
                cell_root = output_dir / f"{attack}_{defense}_seed{seed}"
                summary = _load_summary(cell_root)
                if summary is None:
                    continue
                final = summary.get("final", {}) or {}
                clean.append(float(final.get("clean_acc", float("nan"))))
                asr.append(float(final.get("backdoor_acc", final.get("asr", float("nan")))))
                # ``rl_sim2real_gap`` is per-round in ``series`` (see
                # ``runtime.summaries_to_dict``); attack-level summary has no
                # top-level ``attack_metrics``. Use the post-training tail mean
                # so the gap reflects the deployed policy, not the warmup.
                series = summary.get("series", {}) or {}
                gap = _mean_post_training(series.get("rl_sim2real_gap", []) or [])
                if gap is not None:
                    gaps.append(float(gap))
            rows.append(
                {
                    "attack": attack,
                    "defense": defense,
                    "n_seeds": len(clean),
                    "clean_mean": statistics.fmean(clean) if clean else float("nan"),
                    "clean_std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
                    "asr_mean": statistics.fmean(asr) if asr else float("nan"),
                    "asr_std": statistics.pstdev(asr) if len(asr) > 1 else 0.0,
                    "sim2real_gap_mean": statistics.fmean(gaps) if gaps else float("nan"),
                }
            )
    return rows


def main() -> int:
    args = _parse_args()
    if not args.base_config.is_file():
        sys.exit(f"base config not found: {args.base_config}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_yaml(args.base_config)

    cells_dir = args.output_dir / "_cells"
    for attack in args.attacks:
        for defense in args.defenses:
            for seed in args.seeds:
                cell_args = args
                if (
                    attack == "rl_backdoor"
                    and args.rl_backdoor_separate_frozen_eval
                    and not args.rl_backdoor_policy_checkpoint
                ):
                    train_args = _args_with(
                        args,
                        output_dir=args.output_dir / "_policy_train",
                        tb_root=(args.tb_root / "_policy_train") if args.tb_root else None,
                        rounds=int(args.rl_backdoor_train_rounds),
                        rl_backdoor_policy_checkpoint=None,
                    )
                    train_cfg = _cell_config(
                        base, attack=attack, defense=defense, seed=seed, args=train_args
                    )
                    train_cfg_path = cells_dir / f"{attack}_{defense}_seed{seed}_train.yaml"
                    _dump_yaml(train_cfg, train_cfg_path)
                    if _run_cell(train_cfg_path, dry_run=args.dry_run) != 0:
                        print(f"cell failed: {train_cfg_path}", file=sys.stderr)
                        continue

                    train_cell_root = train_args.output_dir / f"{attack}_{defense}_seed{seed}"
                    checkpoint = _policy_checkpoint_for_cell(train_cell_root)
                    if checkpoint is None:
                        if args.dry_run:
                            checkpoint = train_cell_root / "<run_name>" / "checkpoints" / "rl_policy_latest.pt"
                        else:
                            print(
                                f"missing rl_backdoor policy checkpoint under {train_cell_root}",
                                file=sys.stderr,
                            )
                            continue
                    cell_args = _args_with(args, rl_backdoor_policy_checkpoint=checkpoint)

                cfg = _cell_config(base, attack=attack, defense=defense, seed=seed, args=cell_args)
                cfg_path = cells_dir / f"{attack}_{defense}_seed{seed}.yaml"
                _dump_yaml(cfg, cfg_path)
                if _run_cell(cfg_path, dry_run=args.dry_run) != 0:
                    print(f"cell failed: {cfg_path}", file=sys.stderr)

    rows = _aggregate(args.output_dir, args.attacks, args.defenses, args.seeds)
    out_csv = args.output_dir / "benchmark_summary.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_csv}")
    else:
        print("no cells completed; nothing to aggregate", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
