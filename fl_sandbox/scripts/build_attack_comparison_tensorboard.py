"""Build a TensorBoard view that overlays attackers with shared metric names.

The robust benchmark matrix logs many useful metrics, but some matrix tags
include the defense and attack name. TensorBoard then renders one card per
attacker instead of one card per metric. This postprocess step rewrites the
existing summary.json files so each attacker is a TensorBoard run with the
same scalar tags.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.core.postprocess.tensorboard_utils import build_summary_writer, payload_to_series


CORE_METRICS = (
    ("accuracy/clean", "clean_acc"),
    ("loss/clean", "clean_loss"),
    ("backdoor/accuracy", "backdoor_acc"),
    ("backdoor/asr", "asr"),
    ("updates/benign_norm_mean", "mean_benign_norm"),
    ("updates/malicious_norm_mean", "mean_malicious_norm"),
    ("updates/malicious_cosine_mean", "mean_malicious_cosine"),
    ("time/round_seconds", "round_seconds"),
)

RL_METRIC_PREFIXES = (
    "rl_",
)


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _write_tag_series(writer, tag_series: Mapping[str, Sequence[object]]) -> None:
    max_steps = max((len(values) for values in tag_series.values()), default=0)
    for step in range(1, max_steps + 1):
        for tag in sorted(tag_series):
            values = tag_series[tag]
            if step <= len(values) and _is_finite(values[step - 1]):
                writer.add_scalar(tag, float(values[step - 1]), step)


def _find_summary(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "summary.json").is_file():
        return path / "summary.json"
    matches = sorted(path.glob("*/summary.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one summary.json under {path}, found {len(matches)}")
    return matches[0]


def _load_payload(summary_path: Path) -> dict:
    with summary_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _truncate_payload(payload: dict, round_limit: int | None) -> dict:
    if not round_limit or int(round_limit) <= 0:
        return payload
    limit = int(round_limit)
    truncated = dict(payload)
    if isinstance(payload.get("series"), dict):
        truncated["series"] = {
            key: values[:limit] if isinstance(values, list) else values
            for key, values in payload["series"].items()
        }
    if isinstance(payload.get("rounds"), list):
        truncated["rounds"] = payload["rounds"][:limit]
    return truncated


def _round_counts(payload: Mapping[str, object]) -> tuple[list[int], list[int]]:
    sampled_counts: list[int] = []
    selected_counts: list[int] = []
    for row in payload.get("rounds", []) or []:
        sampled = row.get("sampled_clients", []) if isinstance(row, dict) else []
        selected = row.get("selected_attackers", []) if isinstance(row, dict) else []
        sampled_counts.append(len(sampled) if isinstance(sampled, list) else 0)
        selected_counts.append(len(selected) if isinstance(selected, list) else 0)
    return sampled_counts, selected_counts


def _aligned_delta(reference: Sequence[float], values: Sequence[float], sign: float = 1.0) -> list[float]:
    count = min(len(reference), len(values))
    return [float(sign * (reference[idx] - values[idx])) for idx in range(count)]


def _write_text(writer, tag: str, value: str) -> None:
    try:
        writer.add_text(tag, value, 0)
    except AttributeError:
        return


def _write_run(
    *,
    run_dir: Path,
    attack_name: str,
    payload: dict,
    clean_series: Mapping[str, list[float]],
) -> None:
    writer = build_summary_writer(run_dir)
    series = payload_to_series(payload)
    tag_series: dict[str, Sequence[object]] = {}

    for tag, key in CORE_METRICS:
        tag_series[tag] = series.get(key, [])

    if "round_seconds" in series:
        tag_series["time/elapsed_seconds"] = np.cumsum(series["round_seconds"]).tolist()

    sampled_counts, selected_counts = _round_counts(payload)
    tag_series["sampling/sampled_clients"] = sampled_counts
    tag_series["sampling/selected_attackers"] = selected_counts

    clean_acc = clean_series.get("clean_acc", [])
    clean_loss = clean_series.get("clean_loss", [])
    tag_series["comparison/accuracy_drop_vs_clean"] = _aligned_delta(
        clean_acc,
        series.get("clean_acc", []),
        sign=1.0,
    )
    tag_series["comparison/loss_delta_vs_clean"] = _aligned_delta(
        clean_loss,
        series.get("clean_loss", []),
        sign=-1.0,
    )

    for key, values in sorted(series.items()):
        if key.startswith(RL_METRIC_PREFIXES):
            tag_series[f"rl/{key.removeprefix('rl_')}"] = values

    _write_tag_series(writer, tag_series)

    config = payload.get("config", {})
    _write_text(writer, "run/attack_name", attack_name)
    _write_text(writer, "run/config", json.dumps(config, indent=2, sort_keys=True))
    writer.flush()
    writer.close()


def build_view(
    *,
    summary_root: Path,
    output_root: Path,
    defense: str,
    attacks: Sequence[str],
    clean_attack: str,
    summary_overrides: Mapping[str, Path],
    round_limit: int | None,
    force: bool,
) -> Path:
    defense_root = summary_root / defense
    if not defense_root.is_dir():
        raise FileNotFoundError(f"Defense summary root does not exist: {defense_root}")

    selected_attacks = list(attacks) if attacks else sorted(path.name for path in defense_root.iterdir() if path.is_dir())
    if clean_attack not in selected_attacks:
        selected_attacks.insert(0, clean_attack)

    view_root = output_root / f"{defense}_attackers"
    if force and view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True, exist_ok=True)

    clean_source = summary_overrides.get(clean_attack, defense_root / clean_attack)
    clean_payload = _truncate_payload(_load_payload(_find_summary(clean_source)), round_limit)
    clean_series = payload_to_series(clean_payload)

    for attack_name in selected_attacks:
        source = summary_overrides.get(attack_name, defense_root / attack_name)
        payload = _truncate_payload(_load_payload(_find_summary(source)), round_limit)
        _write_run(
            run_dir=view_root / attack_name,
            attack_name=attack_name,
            payload=payload,
            clean_series=clean_series,
        )

    return view_root


def parse_summary_overrides(specs: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for spec in specs:
        attack_name, sep, path = spec.partition(":")
        if not sep or not attack_name or not path:
            raise ValueError("--summary-override must be formatted as attack_name:/path/to/run_or_summary.json")
        overrides[attack_name] = Path(path)
    return overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("fl_sandbox/outputs/current_benchmark_view/benchmark_outputs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("fl_sandbox/runs/current_benchmark_view/comparisons"),
    )
    parser.add_argument("--defense", default="clipped_median")
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=[
            "clean",
            "ipm",
            "lmp",
            "dba",
            "bfl",
            "rl_clipped_median_scaleaware",
        ],
    )
    parser.add_argument("--clean-attack", default="clean")
    parser.add_argument(
        "--summary-override",
        action="append",
        default=[],
        help="Override one attack source as attack_name:/path/to/run_or_summary.json",
    )
    parser.add_argument(
        "--round-limit",
        type=int,
        default=0,
        help="If positive, truncate all loaded summaries to this many rounds.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    view_root = build_view(
        summary_root=args.summary_root,
        output_root=args.output_root,
        defense=args.defense,
        attacks=args.attacks,
        clean_attack=args.clean_attack,
        summary_overrides=parse_summary_overrides(args.summary_override),
        round_limit=args.round_limit,
        force=args.force,
    )
    print(f"Attack comparison TensorBoard view: {view_root}")


if __name__ == "__main__":
    main(sys.argv[1:])
