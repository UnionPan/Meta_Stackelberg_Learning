"""Whitebox backdoor post-defense selector for a trained Meta-SG checkpoint.

This script is an evaluation-only diagnostic.  It assumes the defender can
validate candidates on the known trigger/target set exposed by the sandbox
backdoor evaluator, then chooses the clean-safe candidate with the lowest ASR.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_sg.scripts import evaluate_meta_sg_direct as direct_eval


@dataclass(frozen=True)
class Candidate:
    name: str
    fixed_neuroclip_epsilon: float | None = None
    post_defense_mode: str = "model_aware_neuroclip"
    fixed_pruning_mask_rate: float | None = None


def parse_eps_candidates(value: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.lower() == "base":
            candidates.append(Candidate(name="base", fixed_neuroclip_epsilon=None))
            continue
        eps = float(item)
        candidates.append(Candidate(name=f"neuroclip_eps_{item}", fixed_neuroclip_epsilon=eps))
    if not candidates:
        raise ValueError("At least one epsilon candidate is required.")
    return candidates


def parse_pruning_candidates(value: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        mask_rate = float(item)
        candidates.append(
            Candidate(
                name=f"pruning_{item}",
                fixed_neuroclip_epsilon=None,
                post_defense_mode="model_aware_pruning",
                fixed_pruning_mask_rate=mask_rate,
            )
        )
    return candidates


def select_whitebox_candidate(records: list[dict], *, clean_floor: float) -> dict:
    if not records:
        raise ValueError("Cannot select from an empty record list.")
    clean_safe = [
        record
        for record in records
        if _finite(record.get("final_clean_acc")) >= float(clean_floor)
    ]
    if clean_safe:
        selected = min(
            clean_safe,
            key=lambda record: (
                _finite(record.get("final_backdoor_acc")),
                -_finite(record.get("final_clean_acc")),
            ),
        )
        result = dict(selected)
        result["selection_reason"] = "lowest_asr_clean_safe"
        return result
    selected = max(records, key=lambda record: _finite(record.get("final_clean_acc")))
    result = dict(selected)
    result["selection_reason"] = "no_clean_safe_candidate"
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scenarios", default="bfl,dba,rl_backdoor")
    parser.add_argument("--eps-candidates", default="base,1,2,4,6,8,10")
    parser.add_argument("--pruning-candidates", default="")
    parser.add_argument("--clean-floor", type=float, default=0.90)
    parser.add_argument("--lambda-bd", type=float, default=1.0)
    parser.add_argument("--H", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=4)
    parser.add_argument("--subsample-rate", type=float, default=0.2)
    parser.add_argument("--client-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl-seed", type=int, default=506)
    parser.add_argument("--defender-third-action", choices=["neuroclip", "server_lr", "both"], default="neuroclip")
    parser.add_argument("--neuroclip-eps-min", type=float, default=1.0)
    parser.add_argument("--neuroclip-eps-max", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    candidates = parse_eps_candidates(args.eps_candidates) + parse_pruning_candidates(args.pruning_candidates)
    scenario_names = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    all_records: list[dict] = []
    selections: list[dict] = []

    for candidate in candidates:
        eval_args = _direct_eval_args(args, candidate)
        direct_args = direct_eval.parse_args(eval_args)
        device = direct_eval._resolve_device(direct_args.device)
        direct_eval.torch.manual_seed(int(direct_args.seed))
        direct_eval.np.random.seed(int(direct_args.seed))
        if device.type == "cuda":
            direct_eval.torch.cuda.manual_seed_all(int(direct_args.seed))
        defender = _load_defender(direct_args, device)
        probe_scenarios = direct_eval._scenarios(direct_args)
        probe_obs = direct_eval._make_common_probe_obs(direct_args, probe_scenarios[0])
        del probe_obs
        for scenario in probe_scenarios:
            record = direct_eval._evaluate_scenario(direct_args, defender, scenario)
            record["candidate"] = candidate.name
            record["fixed_neuroclip_epsilon"] = (
                None if candidate.fixed_neuroclip_epsilon is None else float(candidate.fixed_neuroclip_epsilon)
            )
            record["post_defense_mode"] = candidate.post_defense_mode
            record["fixed_pruning_mask_rate"] = (
                None if candidate.fixed_pruning_mask_rate is None else float(candidate.fixed_pruning_mask_rate)
            )
            all_records.append(record)
            print(
                candidate.name,
                scenario.name,
                "clean=",
                round(float(record["final_clean_acc"]), 4),
                "asr=",
                round(float(record["final_backdoor_acc"]), 4),
            )

    for scenario_name in scenario_names:
        scenario_records = [record for record in all_records if record["scenario"] == scenario_name]
        selected = select_whitebox_candidate(scenario_records, clean_floor=float(args.clean_floor))
        selections.append(selected)
        print(
            "[selected]",
            scenario_name,
            selected["candidate"],
            "clean=",
            round(float(selected["final_clean_acc"]), 4),
            "asr=",
            round(float(selected["final_backdoor_acc"]), 4),
            "reason=",
            selected["selection_reason"],
        )

    output = {
        "config": vars(args),
        "candidates": [candidate.__dict__ for candidate in candidates],
        "records": all_records,
        "selections": selections,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _direct_eval_args(args: argparse.Namespace, candidate: Candidate) -> list[str]:
    eval_args = [
        "--checkpoint",
        str(args.checkpoint),
        "--output-json",
        str(Path(args.output_json).with_suffix(f".{candidate.name}.tmp.json")),
        "--scenario-set",
        "mixed",
        "--scenario-filter",
        str(args.scenarios),
        "--lambda-bd",
        str(args.lambda_bd),
        "--H",
        str(args.H),
        "--num-clients",
        str(args.num_clients),
        "--num-attackers",
        str(args.num_attackers),
        "--subsample-rate",
        str(args.subsample_rate),
        "--client-samples",
        str(args.client_samples),
        "--eval-samples",
        str(args.eval_samples),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--rl-seed",
        str(args.rl_seed),
        "--defender-third-action",
        str(args.defender_third_action),
        "--post-defense-mode",
        candidate.post_defense_mode,
        "--neuroclip-eps-min",
        str(args.neuroclip_eps_min),
        "--neuroclip-eps-max",
        str(args.neuroclip_eps_max),
    ]
    if candidate.fixed_neuroclip_epsilon is not None:
        eval_args.extend(["--fixed-neuroclip-epsilon", str(candidate.fixed_neuroclip_epsilon)])
    if candidate.fixed_pruning_mask_rate is not None:
        eval_args.extend(["--fixed-pruning-mask-rate", str(candidate.fixed_pruning_mask_rate)])
    return eval_args


def _load_defender(args: argparse.Namespace, device):
    probe = direct_eval.FLSandboxCoordinatorAdapter(
        direct_eval._sandbox_config(args, "clean", seed=args.seed, patch={"num_attackers": 0})
    )
    obs_dim = direct_eval.BSMGEnv(
        coordinator=probe,
        attack_type=direct_eval._attack_type("clean"),
        attack_strategy=None,
        defense_strategy=direct_eval.PaperDefenseStrategy(),
        config=direct_eval._bsmg_config(args),
        evaluator=None,
        post_training_evaluator=direct_eval._post_training_evaluator_for_args(args, probe),
    ).reset(seed=args.seed).shape[0]
    defender = direct_eval.TD3Agent(
        obs_dim,
        direct_eval._defender_action_dim(args),
        direct_eval.TD3Config(hidden_dim=args.hidden_dim, batch_size=16, buffer_capacity=4096, warmup_steps=0),
        device=device,
    )
    defender.load(direct_eval._checkpoint_path(args.checkpoint))
    return defender


def _finite(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return result if math.isfinite(result) else float("-inf")


if __name__ == "__main__":
    main()
