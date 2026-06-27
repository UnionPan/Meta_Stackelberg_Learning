"""Select Meta-SG checkpoints from direct-evaluation JSON files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def summarize_eval_file(path: str | Path) -> dict:
    eval_path = Path(path)
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{eval_path} must contain a non-empty list of scenario records")

    clean = [_finite_float(row.get("final_clean_acc")) for row in rows]
    backdoor = [_finite_float(row.get("final_backdoor_acc")) for row in rows]
    score = [
        _finite_float(row.get("final_defense_score", row.get("final_defender_reward")))
        for row in rows
    ]
    scenarios = [str(row.get("scenario", row.get("attack_type", ""))) for row in rows]
    targeted_names = {"bfl", "dba", "rl_backdoor"}
    per_attack = {}
    attack_scores = []
    targeted_backdoor = []
    for scenario, c, b, s in zip(scenarios, clean, backdoor, score):
        targeted = scenario in targeted_names
        per_attack[scenario] = {
            "clean": c,
            "backdoor": b,
            "score": s,
            "targeted": targeted,
        }
        if scenario != "clean":
            attack_scores.append(s)
        if targeted:
            targeted_backdoor.append(b)
    return {
        "path": str(eval_path),
        "n": len(rows),
        "scenarios": scenarios,
        "mean_clean": _mean(clean),
        "worst_clean": min(clean),
        "mean_score": _mean(score),
        "worst_attack_score": min(attack_scores or score),
        "max_backdoor": max(backdoor),
        "max_targeted_backdoor": max(targeted_backdoor or backdoor),
        "per_attack": per_attack,
    }


def select_checkpoint(
    paths: Iterable[str | Path],
    *,
    metric: str = "mean_score",
    clean_floor: float | None = None,
    clean_penalty_weight: float = 1.0,
    backdoor_ceiling: float | None = None,
) -> dict:
    summaries = [summarize_eval_file(path) for path in paths]
    if not summaries:
        raise ValueError("At least one evaluation JSON path is required")

    eligible = list(summaries)
    if metric == "clean_floor_then_score":
        if clean_floor is None:
            raise ValueError("clean_floor_then_score requires clean_floor")
        eligible = [row for row in summaries if row["worst_clean"] >= float(clean_floor)]
        ranking_pool = eligible or summaries
        selected = max(ranking_pool, key=lambda row: (row["mean_score"], row["worst_clean"]))
    elif metric == "mean_score":
        selected = max(summaries, key=lambda row: (row["mean_score"], row["worst_clean"]))
    elif metric == "worst_clean":
        selected = max(summaries, key=lambda row: (row["worst_clean"], row["mean_score"]))
    elif metric == "score_with_clean_penalty":
        floor = float(clean_floor or 0.0)

        def penalized(row: dict) -> tuple[float, float]:
            violation = max(0.0, floor - row["worst_clean"])
            adjusted = row["mean_score"] - float(clean_penalty_weight) * violation
            return adjusted, row["worst_clean"]

        selected = max(summaries, key=penalized)
    elif metric == "backdoor_ceiling_then_score":
        if clean_floor is None:
            raise ValueError("backdoor_ceiling_then_score requires clean_floor")
        if backdoor_ceiling is None:
            raise ValueError("backdoor_ceiling_then_score requires backdoor_ceiling")
        eligible = [
            row
            for row in summaries
            if row["worst_clean"] >= float(clean_floor)
            and row["max_targeted_backdoor"] <= float(backdoor_ceiling)
        ]
        ranking_pool = eligible or summaries
        selected = max(
            ranking_pool,
            key=lambda row: (
                row["mean_score"],
                row["worst_attack_score"],
                row["worst_clean"],
                -row["max_targeted_backdoor"],
            ),
        )
    else:
        raise ValueError(f"Unsupported selection metric: {metric}")

    return {
        "metric": metric,
        "clean_floor": clean_floor,
        "clean_penalty_weight": float(clean_penalty_weight),
        "backdoor_ceiling": backdoor_ceiling,
        "summaries": summaries,
        "eligible": eligible,
        "selected": selected,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", nargs="+", help="Direct-evaluation JSON files to compare.")
    parser.add_argument(
        "--metric",
        choices=[
            "mean_score",
            "worst_clean",
            "clean_floor_then_score",
            "score_with_clean_penalty",
            "backdoor_ceiling_then_score",
        ],
        default="mean_score",
    )
    parser.add_argument("--clean-floor", type=float, default=None)
    parser.add_argument("--clean-penalty-weight", type=float, default=1.0)
    parser.add_argument("--backdoor-ceiling", type=float, default=None)
    parser.add_argument("--output-json", default="")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    report = select_checkpoint(
        args.eval_json,
        metric=args.metric,
        clean_floor=args.clean_floor,
        clean_penalty_weight=args.clean_penalty_weight,
        backdoor_ceiling=args.backdoor_ceiling,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return report


def _finite_float(value) -> float:
    result = float(value)
    if math.isnan(result):
        raise ValueError("Evaluation metric cannot be NaN")
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric list")
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
