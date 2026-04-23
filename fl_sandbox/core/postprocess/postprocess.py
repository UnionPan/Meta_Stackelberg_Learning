"""Unified postprocess entry point for clean and attack run exports."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_sandbox.core.experiment_builders import split_suffix
from fl_sandbox.core.postprocess.tensorboard_utils import (
    build_summary_writer,
    coerce_series,
    payload_to_series,
    write_client_stat_scalars,
    write_standard_tensorboard_run,
)


@dataclass
class LoadedRun:
    input_dir: Path
    run_name: str
    config: dict
    series: dict[str, list[float]]
    payload: dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified postprocess for clean runs and multi-method attack export"
    )
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--clean_input_dir", type=str, default="")
    parser.add_argument("--attack_input_dir", type=str, nargs="*", default=[])
    parser.add_argument("--attack_root", type=str, default="")
    parser.add_argument("--methods", type=str, nargs="*", default=[])
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--skip_clean_write", action="store_true")
    return parser.parse_args(argv)


def build_postprocess_hint_lines(
    *,
    attack_type: str,
    defense_type: str,
    split_mode: str,
    noniid_q: float,
    output_dir: Path,
    tb_dir: Path,
) -> list[str]:
    if attack_type == "clean":
        return [
            "Run postprocess with: "
            f"python fl_sandbox/core/postprocess/postprocess.py --input_dir {output_dir} "
            f"--tb_dir {tb_dir}"
        ]

    suffix = split_suffix(split_mode, noniid_q)
    suggested_clean_dir = Path(f"fl_sandbox/outputs/clean_{defense_type}_{suffix}_benchmark")
    return [
        "Run postprocess with: "
        "python fl_sandbox/core/postprocess/postprocess.py "
        f"--clean_input_dir {suggested_clean_dir} "
        f"--attack_input_dir {output_dir} --tb_dir {tb_dir}_compare"
    ]


def _resolve_summary_path(input_dir: Path, label: str) -> Path:
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        return summary_path

    parent = input_dir.parent if input_dir.parent.exists() else None
    available = []
    if parent is not None:
        available = sorted(
            candidate.name
            for candidate in parent.iterdir()
            if candidate.is_dir() and (candidate / "summary.json").exists()
        )

    message_lines = [f"Missing {label} summary: {summary_path}"]
    if available:
        similar = difflib.get_close_matches(input_dir.name, available, n=5, cutoff=0.0)
        if similar:
            message_lines.append("Available directories with summary.json:")
            message_lines.extend(f"  - {name}" for name in similar)
    raise FileNotFoundError("\n".join(message_lines))


def _load_run(input_dir: Path, *, label: str, preferred_run: str | None = None) -> LoadedRun:
    with _resolve_summary_path(input_dir, label).open(encoding="utf-8") as fh:
        payload = json.load(fh)

    config = payload.get("config", {})
    run_name = preferred_run or payload.get("attack_type") or config.get("attack_type")

    if "series" in payload:
        return LoadedRun(
            input_dir=input_dir,
            run_name=run_name or label,
            config=config,
            series=coerce_series(payload["series"]),
            payload=payload,
        )

    if "runs" in payload:
        if preferred_run and preferred_run in payload["runs"]:
            run_name = preferred_run
        elif run_name and run_name in payload["runs"]:
            pass
        elif label == "clean" and "clean" in payload["runs"]:
            run_name = "clean"
        else:
            attack_keys = [key for key in payload["runs"] if key != "clean"]
            if len(attack_keys) == 1:
                run_name = attack_keys[0]
        if not run_name or run_name not in payload["runs"]:
            raise ValueError(f"Could not resolve run '{preferred_run or label}' in {input_dir / 'summary.json'}")
        return LoadedRun(
            input_dir=input_dir,
            run_name=run_name,
            config=config,
            series=coerce_series(payload["runs"][run_name]),
            payload=payload,
        )

    return LoadedRun(
        input_dir=input_dir,
        run_name=run_name or "clean",
        config=config,
        series=payload_to_series(payload),
        payload=payload,
    )


def _reset_tb_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _resolve_attack_inputs(args: argparse.Namespace) -> list[Path]:
    if args.attack_input_dir:
        return [Path(path) for path in args.attack_input_dir]

    if not args.attack_root or not args.methods:
        return []

    attack_root = Path(args.attack_root)
    candidates = [
        candidate
        for candidate in sorted(attack_root.iterdir())
        if candidate.is_dir() and (candidate / "summary.json").exists()
    ]

    resolved: list[Path] = []
    for method in args.methods:
        matched = None
        for candidate in candidates:
            try:
                run = _load_run(candidate, label="attack")
            except Exception:
                continue
            if run.run_name == method:
                matched = candidate
                break
        if matched is None:
            for candidate in candidates:
                if f"_{method}_" in candidate.name or candidate.name.startswith(f"{method}_") or candidate.name.endswith(f"_{method}"):
                    matched = candidate
                    break
        if matched is None:
            raise FileNotFoundError(f"Could not find attack output for method '{method}' under {attack_root}")
        resolved.append(matched)
    return resolved


def _run_clean_postprocess(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"
    client_metrics_path = input_dir / "client_metrics.csv"

    with summary_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    with client_metrics_path.open(encoding="utf-8", newline="") as fh:
        client_rows = list(csv.DictReader(fh))

    clean_series = payload_to_series(payload)
    if args.tb_dir:
        writer = build_summary_writer(Path(args.tb_dir))
        write_standard_tensorboard_run(
            writer,
            "clean",
            payload.get("config", {}),
            clean_series,
            include_attack=False,
            total_seconds=float(payload.get("total_seconds", 0.0)) or None,
        )
        write_client_stat_scalars(writer, client_rows, round_count=len(payload["rounds"]))
        writer.flush()
        writer.close()

    print(f"Postprocess finished for: {input_dir}")
    if args.tb_dir:
        print(f"TensorBoard run written to: {args.tb_dir}")
    else:
        print("No --tb_dir provided; no TensorBoard data was written.")


def _run_compare_postprocess(args: argparse.Namespace) -> None:
    clean_input_dir = Path(args.clean_input_dir)
    clean_run = _load_run(clean_input_dir, label="clean", preferred_run="clean")
    attack_inputs = _resolve_attack_inputs(args)
    if not attack_inputs:
        raise SystemExit(
            "Attack export mode requires attack inputs. Pass --attack_input_dir ... or --attack_root ... --methods ..."
        )

    attack_runs = [_load_run(path, label="attack") for path in attack_inputs]
    attack_names = [run.run_name for run in attack_runs]
    duplicate_names = sorted({name for name in attack_names if attack_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate attack names detected: {', '.join(duplicate_names)}")

    expected_rounds = len(clean_run.series["clean_acc"])
    for attack_run in attack_runs:
        if len(attack_run.series["clean_acc"]) != expected_rounds:
            raise ValueError(
                f"Round mismatch between clean ({expected_rounds}) and "
                f"{attack_run.run_name} ({len(attack_run.series['clean_acc'])})."
            )

    if args.tb_dir:
        tb_dir = Path(args.tb_dir)
        clean_run_dir = tb_dir / "clean"
        if not args.skip_clean_write:
            _reset_tb_run_dir(clean_run_dir)
        for attack_run in attack_runs:
            _reset_tb_run_dir(tb_dir / attack_run.run_name)

        if not args.skip_clean_write:
            clean_writer = build_summary_writer(clean_run_dir)
            write_standard_tensorboard_run(
                clean_writer,
                "clean",
                clean_run.config,
                clean_run.series,
                include_attack=False,
                total_seconds=float(clean_run.payload.get("total_seconds", 0.0)) or None,
            )
            clean_writer.flush()
            clean_writer.close()

        for attack_run in attack_runs:
            attack_writer = build_summary_writer(tb_dir / attack_run.run_name)
            write_standard_tensorboard_run(
                attack_writer,
                attack_run.run_name,
                attack_run.config,
                attack_run.series,
                include_attack=True,
                total_seconds=float(attack_run.payload.get("total_seconds", 0.0)) or None,
            )
            attack_writer.flush()
            attack_writer.close()

    print(f"Postprocess finished for clean: {clean_input_dir}")
    print(f"Postprocess finished for attacks: {', '.join(attack_names)}")
    if args.tb_dir:
        print(f"TensorBoard run written to: {Path(args.tb_dir) / 'clean'}")
        for attack_name in attack_names:
            print(f"TensorBoard run written to: {Path(args.tb_dir) / attack_name}")
    else:
        print("No --tb_dir provided; no TensorBoard data was written.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_dir:
        _run_clean_postprocess(args)
        return
    if args.clean_input_dir:
        _run_compare_postprocess(args)
        return
    raise SystemExit("Pass either --input_dir for clean mode or --clean_input_dir for compare mode.")


if __name__ == "__main__":
    main()
