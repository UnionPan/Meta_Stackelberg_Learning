"""Application-facing experiment service.

This module accepts the structured ``RunConfig`` used by the new benchmark
layout while delegating persistence-compatible execution to the historical core
service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fl_sandbox.config import RunConfig, config_to_namespace
from fl_sandbox.core.experiment_builders import build_run_name
from fl_sandbox.core.experiment_service import (
    ExperimentRunResult,
    completion_lines,
    execute_experiment as execute_core_experiment,
    persist_experiment_artifacts,
)


def prepare_run_args(run_config: RunConfig):
    """Build legacy argparse-style args from structured config."""
    run_args = config_to_namespace(run_config)
    run_name = build_run_name(
        dataset=run_args.dataset,
        attack_type=run_args.attack_type,
        defense_type=run_args.defense_type,
        split_mode=run_args.split_mode,
        noniid_q=run_args.noniid_q,
        rounds=run_args.rounds,
    )
    run_args.output_dir = str(Path(run_args.output_root) / run_name)
    run_args.tb_dir = str(Path(run_args.tb_root) / run_name)
    return run_args, run_name


def execute_experiment(
    run_config_or_args: RunConfig | Any,
    *,
    progress_desc: str | None = None,
    run_config: RunConfig | None = None,
) -> ExperimentRunResult:
    """Execute one experiment from either new structured config or legacy args."""
    if isinstance(run_config_or_args, RunConfig):
        resolved_config = run_config_or_args.normalize()
        run_args, run_name = prepare_run_args(resolved_config)
        return execute_core_experiment(
            run_args,
            progress_desc=progress_desc or run_name,
            run_config=resolved_config,
        )
    return execute_core_experiment(
        run_config_or_args,
        progress_desc=progress_desc,
        run_config=run_config,
    )


__all__ = [
    "ExperimentRunResult",
    "completion_lines",
    "execute_experiment",
    "persist_experiment_artifacts",
    "prepare_run_args",
]
