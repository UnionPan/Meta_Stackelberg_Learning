from __future__ import annotations

import ast
from pathlib import Path

from fl_sandbox.config.schema import DefenderSection, RunConfig
from fl_sandbox.core.experiment_builders import build_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_level_defender_imports_and_factory():
    from fl_sandbox.defenders import (
        DEFENSE_CHOICES,
        ClippedMedianDefender,
        FLTrustDefender,
        FedAvgDefender,
        GeometricMedianDefender,
        KrumDefender,
        MedianDefender,
        MultiKrumDefender,
        PaperNormTrimmedMeanDefender,
        SandboxDefender,
        TrimmedMeanDefender,
        build_defender_config_kwargs,
        create_defender,
        supported_defense_types,
    )

    assert DEFENSE_CHOICES == supported_defense_types()
    assert "paper_norm_trimmed_mean" in DEFENSE_CHOICES
    assert issubclass(FedAvgDefender, SandboxDefender)
    assert issubclass(KrumDefender, SandboxDefender)
    assert issubclass(MultiKrumDefender, SandboxDefender)
    assert issubclass(MedianDefender, SandboxDefender)
    assert issubclass(ClippedMedianDefender, SandboxDefender)
    assert issubclass(GeometricMedianDefender, SandboxDefender)
    assert issubclass(TrimmedMeanDefender, SandboxDefender)
    assert issubclass(PaperNormTrimmedMeanDefender, SandboxDefender)
    assert issubclass(FLTrustDefender, SandboxDefender)

    defender = create_defender(
        DefenderSection(
            type="paper_norm_trimmed_mean",
            clipped_median_norm=1.25,
            trimmed_mean_ratio=0.15,
        )
    )
    assert isinstance(defender, PaperNormTrimmedMeanDefender)
    assert build_defender_config_kwargs(
        DefenderSection(
            type="paper_norm_trimmed_mean",
            clipped_median_norm=1.25,
            trimmed_mean_ratio=0.15,
        )
    )["defense_type"] == "paper_norm_trimmed_mean"


def test_paper_norm_trimmed_mean_builds_through_run_config():
    config = build_config(
        RunConfig.from_flat_dict(
            {
                "defense_type": "paper_norm_trimmed_mean",
                "clipped_median_norm": 1.25,
                "trimmed_mean_ratio": 0.15,
            }
        )
    )

    assert config.defense_type == "paper_norm_trimmed_mean"
    assert config.clipped_median_norm == 1.25
    assert config.trimmed_mean_ratio == 0.15


def test_no_old_defender_import_paths_remain_in_python_sources():
    allowed_files = {
        Path("tests/test_fl_sandbox_defenders_public_api.py"),
    }
    source_roots = (
        PROJECT_ROOT / "fl_sandbox",
        PROJECT_ROOT / "meta_sg",
        PROJECT_ROOT / "tests",
    )

    offenders: list[tuple[str, str]] = []
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            rel_path = path.relative_to(PROJECT_ROOT)
            if rel_path in allowed_files or "__pycache__" in rel_path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("fl_sandbox.core.defender"):
                            offenders.append((str(rel_path), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("fl_sandbox.core.defender"):
                        offenders.append((str(rel_path), node.module))
                    if rel_path == Path("fl_sandbox/core/experiment_builders.py") and node.level == 1 and node.module == "defender":
                        offenders.append((str(rel_path), "from .defender"))

            if rel_path == Path("fl_sandbox/core/__init__.py") and '".defender"' in path.read_text(encoding="utf-8"):
                offenders.append((str(rel_path), '".defender" lazy export'))

    assert offenders == []
