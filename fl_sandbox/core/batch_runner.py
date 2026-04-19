"""Shared batch-run helpers for sandbox experiment entry points."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .experiment_service import ExperimentRunResult, execute_experiment


@dataclass
class BatchRunRequest:
    run_name: str
    args: argparse.Namespace
    progress_desc: str


def clone_args(base_args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = dict(vars(base_args))
    values.update(overrides)
    return argparse.Namespace(**values)


def execute_batch_run(
    request: BatchRunRequest,
    *,
    payload_transform: Callable[[ExperimentRunResult], dict[str, object]] | None = None,
    artifact_writer: Callable[[ExperimentRunResult, dict[str, object]], None] | None = None,
) -> tuple[ExperimentRunResult, dict[str, object]]:
    result = execute_experiment(request.args, progress_desc=request.progress_desc)
    payload = result.payload if payload_transform is None else payload_transform(result)
    if artifact_writer is not None:
        artifact_writer(result, payload)
    return result, payload


def write_batch_results(output_root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    csv_path = output_root / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_root / "results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return csv_path, json_path
