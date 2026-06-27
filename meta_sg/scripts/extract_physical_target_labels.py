"""Extract physical-target oracle labels from direct-eval JSON artifacts."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meta_sg.scripts.evaluate_meta_sg_direct import _select_physical_target_with_deployment_records


TARGETED_ATTACKS = {"bfl", "dba", "rl_backdoor"}
_PHYSICAL_LABEL_RE = re.compile(r"^physical_a(?P<alpha>[0-9.]+)_b(?P<beta>[0-9.]+)(?:_(?P<suffix>.+))?$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        action="append",
        required=True,
        help="Direct-eval JSON artifact. Can be passed multiple times.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--asr-reduction-margin", type=float, default=0.005)
    parser.add_argument("--query-clean-floor", type=_optional_float, default=0.0)
    parser.add_argument("--query-clean-drop-tolerance", type=_optional_float, default=0.04)
    parser.add_argument("--score-slack", type=float, default=0.0)
    parser.add_argument("--deployment-clean-floor", type=_optional_float, default=None)
    parser.add_argument("--deployment-clean-drop-tolerance", type=_optional_float, default=None)
    parser.add_argument("--deployment-asr-ceiling", type=_optional_float, default=None)
    return parser.parse_args(argv)


def extract_label_samples(
    input_jsons: list[str | Path],
    *,
    asr_reduction_margin: float,
    query_clean_floor: float | None,
    query_clean_drop_tolerance: float | None,
    score_slack: float,
    deployment_clean_floor: float | None,
    deployment_clean_drop_tolerance: float | None,
    deployment_asr_ceiling: float | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    constraints = {
        "asr_reduction_margin": _json_safe(asr_reduction_margin),
        "query_clean_floor": _json_safe(query_clean_floor),
        "query_clean_drop_tolerance": _json_safe(query_clean_drop_tolerance),
        "score_slack": _json_safe(score_slack),
        "deployment_clean_floor": _json_safe(deployment_clean_floor),
        "deployment_clean_drop_tolerance": _json_safe(deployment_clean_drop_tolerance),
        "deployment_asr_ceiling": _json_safe(deployment_asr_ceiling),
    }
    for input_json in input_jsons:
        path = Path(input_json)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            candidates = list(row.get("few_shot_adaptation", {}).get("candidate_scores", []))
            if not candidates:
                continue
            selected, selection = _select_physical_target_with_deployment_records(
                candidates,
                asr_reduction_margin=float(asr_reduction_margin),
                clean_floor=query_clean_floor,
                clean_drop_tolerance=query_clean_drop_tolerance,
                score_slack=float(score_slack),
                deployment_clean_floor=deployment_clean_floor,
                deployment_clean_drop_tolerance=deployment_clean_drop_tolerance,
                deployment_asr_ceiling=deployment_asr_ceiling,
            )
            scenario = str(row.get("scenario", row.get("attack_type", "unknown")))
            attack_name = str(row.get("attack_type", scenario))
            sample = {
                "source_json": str(path),
                "scenario": scenario,
                "attack_name": attack_name,
                "attack_objective": "targeted" if attack_name in TARGETED_ATTACKS else "untargeted",
                "seed": int(row.get("seed", -1)),
                "label": _label_from_offset_label(str(selected.get("offset_label", "zero"))),
                "selected_offset_label": str(selected.get("offset_label", "zero")),
                "selected_offset": _json_safe(selected.get("offset", [])),
                "selected_target_alpha": _json_safe(selected.get("target_alpha")),
                "selected_target_beta": _json_safe(selected.get("target_beta")),
                "selected_target_start_round": _json_safe(selected.get("target_start_round")),
                "selected_target_end_round": _json_safe(selected.get("target_end_round")),
                "accepted": bool(selection.get("accepted", False)),
                "constraints": constraints,
                "base_metrics": _base_metrics(row),
                "oracle_metrics": _oracle_metrics(selection),
                "candidate_query_summaries": [_candidate_summary(item) for item in candidates],
                "selection": _json_safe(selection),
            }
            samples.append(_json_safe(sample))
    return samples


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(sample.get("label", "")) for sample in samples)
    scenario_counts = Counter(str(sample.get("scenario", "")) for sample in samples)
    attack_counts = Counter(str(sample.get("attack_name", "")) for sample in samples)
    return {
        "num_samples": len(samples),
        "label_counts": dict(sorted(label_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "attack_counts": dict(sorted(attack_counts.items())),
        "accepted_count": int(sum(1 for sample in samples if bool(sample.get("accepted", False)))),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    samples = extract_label_samples(
        [Path(item) for item in args.input_json],
        asr_reduction_margin=float(args.asr_reduction_margin),
        query_clean_floor=args.query_clean_floor,
        query_clean_drop_tolerance=args.query_clean_drop_tolerance,
        score_slack=float(args.score_slack),
        deployment_clean_floor=args.deployment_clean_floor,
        deployment_clean_drop_tolerance=args.deployment_clean_drop_tolerance,
        deployment_asr_ceiling=args.deployment_asr_ceiling,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
    metrics = summarize_samples(samples)
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))


def _label_from_offset_label(offset_label: str) -> str:
    if offset_label == "zero":
        return "zero"
    match = _PHYSICAL_LABEL_RE.match(offset_label)
    if not match:
        return offset_label
    alpha = float(match.group("alpha"))
    suffix = match.group("suffix") or "full"
    if suffix.startswith("w"):
        parts = suffix[1:].split("_")
        suffix_label = f"w{parts[-1]}" if parts and parts[-1] else suffix
    else:
        suffix_label = suffix
    return f"a{alpha:.2f}_{suffix_label}"


def _base_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clean_acc": _finite(row.get("final_clean_acc")),
        "backdoor_acc": _finite(row.get("final_backdoor_acc")),
        "defense_score": _finite(row.get("final_defense_score")),
    }


def _oracle_metrics(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_deployment_clean_acc": _finite(selection.get("base_deployment_clean_acc")),
        "deployment_clean_acc": _finite(selection.get("deployment_clean_acc")),
        "deployment_clean_drop": _finite(selection.get("deployment_clean_drop")),
        "clean_drop": _finite(selection.get("deployment_clean_drop")),
        "deployment_backdoor_acc": _finite(selection.get("deployment_backdoor_acc")),
        "deployment_defense_score": _finite(selection.get("deployment_defense_score")),
        "query_selected_offset_label": str(selection.get("query_selected_offset_label", "base")),
        "query_eligible_count": int(selection.get("query_eligible_count", 0)),
        "deployment_eligible_count": int(selection.get("deployment_eligible_count", 0)),
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "offset_label": str(candidate.get("offset_label", "")),
        "label": _label_from_offset_label(str(candidate.get("offset_label", ""))),
        "target_alpha": _json_safe(candidate.get("target_alpha")),
        "target_beta": _json_safe(candidate.get("target_beta")),
        "target_start_round": _json_safe(candidate.get("target_start_round")),
        "target_end_round": _json_safe(candidate.get("target_end_round")),
        "query_clean_mean": _finite(candidate.get("query_clean_mean")),
        "query_backdoor_mean": _finite(candidate.get("query_backdoor_mean")),
        "query_score_mean": _finite(candidate.get("query_score_mean")),
        "deployment_clean_acc": _finite(candidate.get("deployment_clean_acc")),
        "deployment_backdoor_acc": _finite(candidate.get("deployment_backdoor_acc")),
        "deployment_defense_score": _finite(candidate.get("deployment_defense_score")),
    }


def _optional_float(value: str) -> float | None:
    if str(value).lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
