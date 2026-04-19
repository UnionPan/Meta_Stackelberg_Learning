"""Load and merge structured run configuration from YAML and CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import yaml

from .schema import RunConfig


def load_run_config(config_path: str | None = None) -> RunConfig:
    if not config_path:
        return RunConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return RunConfig.from_mapping(payload)


def merge_cli_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    flat = config.to_flat_dict()
    overrides = {key: value for key, value in vars(args).items() if key != "config"}
    flat.update(overrides)
    return RunConfig.from_flat_dict(flat).normalize()


def config_to_namespace(config: RunConfig) -> argparse.Namespace:
    return argparse.Namespace(**config.normalize().to_flat_dict())
