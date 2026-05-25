"""Run benchmark matrices defined by YAML presets."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from fl_sandbox.config import RunConfig
from fl_sandbox.experiments.builders import build_run_name
from fl_sandbox.experiments.service import completion_lines, execute_experiment, persist_experiment_artifacts


def _preset_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark preset must contain a mapping: {path}")
    return payload


def _run_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "benchmark"}


def _matrix_entries(payload: dict) -> Iterable[tuple[str, str]]:
    matrix = payload.get("benchmark", {}).get("matrix", [])
    for entry in matrix:
        for attack in entry.get("attacks", []):
            for defense in entry.get("defenses", []):
                yield str(attack), str(defense)


def benchmark_run_configs(preset_path: Path) -> list[RunConfig]:
    payload = _preset_payload(preset_path)
    base_payload = _run_payload(payload)
    configs: list[RunConfig] = []
    for attack, defense in _matrix_entries(payload):
        run_payload = deepcopy(base_payload)
        run_payload.setdefault("attacker", {})["type"] = attack
        run_payload.setdefault("defender", {})["type"] = defense
        configs.append(RunConfig.from_mapping(run_payload))
    return configs


def run_name_for_config(config: RunConfig) -> str:
    return build_run_name(
        dataset=config.data.dataset,
        attack_type=config.attacker.type,
        defense_type=config.defender.type,
        split_mode=config.data.split_mode,
        noniid_q=config.data.noniid_q,
        rounds=config.runtime.rounds,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix from a YAML preset")
    parser.add_argument("--config", required=True, help="Path to a benchmark preset YAML")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for config in benchmark_run_configs(Path(args.config)):
        run_name = run_name_for_config(config)
        output_dir = Path(config.output.output_root) / run_name
        tb_dir = Path(config.output.tb_root) / run_name
        result = execute_experiment(config, progress_desc=run_name, output_dir=output_dir, tb_dir=tb_dir)
        persist_experiment_artifacts(result, write_tensorboard=False, write_round_metrics=True)
        for line in completion_lines(result):
            print(line)


if __name__ == "__main__":
    main()
