import json
from unittest.mock import Mock

import pytest

from meta_sg.learning import memory_maintenance as mm


def test_memory_maintenance_collects_and_trims(monkeypatch):
    rss_values = iter([20_000, 12_000])
    monkeypatch.setattr(mm, "_read_rss_kib", lambda: next(rss_values))
    monkeypatch.setattr(mm.gc, "collect", lambda: 17)
    monkeypatch.setattr(mm, "_malloc_trim", lambda: (True, True, None))

    result = mm.perform_memory_maintenance()

    assert result.objects_collected == 17
    assert result.rss_before_kib == 20_000
    assert result.rss_after_kib == 12_000
    assert result.rss_released_kib == 8_000
    assert result.malloc_trim_supported is True
    assert result.malloc_trim_succeeded is True
    assert result.warning is None
    assert result.elapsed_seconds >= 0.0
    assert result.as_dict()["rss_released_kib"] == 8_000


def test_memory_maintenance_degrades_when_trim_is_unsupported(monkeypatch):
    monkeypatch.setattr(mm, "_read_rss_kib", lambda: None)
    monkeypatch.setattr(mm.gc, "collect", lambda: 0)
    monkeypatch.setattr(
        mm,
        "_malloc_trim",
        lambda: (False, False, "unsupported platform"),
    )

    result = mm.perform_memory_maintenance()

    assert result.rss_before_kib is None
    assert result.rss_after_kib is None
    assert result.rss_released_kib is None
    assert result.malloc_trim_supported is False
    assert result.malloc_trim_succeeded is False
    assert result.warning == "unsupported platform"


def test_malloc_trim_symbol_failure_is_bounded(monkeypatch):
    monkeypatch.setattr(mm.platform, "system", lambda: "Linux")

    def missing_libc(*_args, **_kwargs):
        raise OSError("missing")

    monkeypatch.setattr(mm.ctypes, "CDLL", missing_libc)

    supported, succeeded, warning = mm._malloc_trim()

    assert supported is False
    assert succeeded is False
    assert warning is not None
    assert len(warning) <= 240


def test_pretraining_memory_maintenance_defaults_off():
    from meta_sg.scripts.run_meta_sg_pretraining import parse_args

    assert parse_args([]).memory_maintenance == "off"
    assert parse_args(["--memory-maintenance", "task"]).memory_maintenance == "task"


def test_pretraining_runs_memory_maintenance_once_per_task(tmp_path, monkeypatch):
    from meta_sg.scripts import run_meta_sg_pretraining as script

    maintenance = Mock(
        side_effect=[
            mm.MemoryMaintenanceResult(
                elapsed_seconds=0.01,
                objects_collected=2,
                rss_before_kib=10_000,
                rss_after_kib=9_000,
                rss_released_kib=1_000,
                malloc_trim_supported=True,
                malloc_trim_succeeded=True,
                warning=None,
            ),
            mm.MemoryMaintenanceResult(
                elapsed_seconds=0.02,
                objects_collected=4,
                rss_before_kib=9_000,
                rss_after_kib=6_928,
                rss_released_kib=2_072,
                malloc_trim_supported=True,
                malloc_trim_succeeded=True,
                warning=None,
            ),
        ]
    )
    monkeypatch.setattr(script, "perform_memory_maintenance", maintenance)

    script.main(
        [
            "--backend",
            "stub",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "run",
            "--T",
            "1",
            "--K",
            "2",
            "--H",
            "1",
            "--l",
            "1",
            "--N-A",
            "1",
            "--post-br-defender-updates",
            "0",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--num-clients",
            "6",
            "--num-attackers",
            "1",
            "--subsample-rate",
            "1.0",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--memory-maintenance",
            "task",
        ]
    )

    assert maintenance.call_count == 2
    record = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0])
    assert len(record["task_records"]) == 2
    assert all("memory_maintenance" in task for task in record["task_records"])
    assert record["memory_maintenance"]["calls"] == 2
    assert record["memory_maintenance"]["objects_collected"] == 6
    assert record["memory_maintenance"]["rss_released_kib"] == 3_072
    assert record["memory_maintenance"]["elapsed_seconds"] == pytest.approx(0.03)
    json.dumps(record, allow_nan=False)
