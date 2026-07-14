"""Write reproducible experiment provenance and atomic stage-status artifacts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return value


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_provenance(repo_root: Path) -> dict:
    dirty_paths = [
        line[3:]
        for line in _git(repo_root, "status", "--short", "--untracked-files=all").splitlines()
        if len(line) >= 4
    ]
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _torch_provenance(device: str) -> dict:
    try:
        import torch
    except Exception as error:  # pragma: no cover - project environment has torch
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    cuda_available = bool(torch.cuda.is_available())
    gpu_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": str(device),
        "gpu_devices": gpu_devices,
    }


def _config_pairs(values: list[str]) -> dict:
    config = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--config must be KEY=VALUE, got {value!r}")
        config[key.strip()] = item
    return config


def write_provenance(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    relevant_environment = {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG",
        )
        if key in os.environ
    }
    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "repo_root": str(repo_root),
        "git": _git_provenance(repo_root),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch": _torch_provenance(args.device),
        },
        "seeds": {
            "master": int(args.master_seed),
            "training": int(args.training_seed),
            "evaluation": int(args.evaluation_seed),
        },
        "device": str(args.device),
        "configuration": _config_pairs(args.config),
        "environment": relevant_environment,
        "artifact_command": [str(item) for item in sys.argv],
    }
    _atomic_write_json(Path(args.output), payload)


def write_status(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
    else:
        prior = {"schema_version": 1, "history": []}

    timestamp = _utc_now()
    entry = {
        "stage": str(args.stage),
        "message": str(args.message),
        "timestamp": timestamp,
    }
    if args.exit_code is not None:
        entry["exit_code"] = int(args.exit_code)
    if args.last_completed_iteration is not None:
        entry["last_completed_iteration"] = int(args.last_completed_iteration)

    history = list(prior.get("history") or [])
    history.append(entry)
    status = {
        **prior,
        "schema_version": 1,
        "stage": entry["stage"],
        "message": entry["message"],
        "updated_at": timestamp,
        "history": history,
    }
    if "exit_code" in entry:
        status["exit_code"] = entry["exit_code"]
    if "last_completed_iteration" in entry:
        status["last_completed_iteration"] = entry["last_completed_iteration"]
    _atomic_write_json(output, status)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    provenance = subparsers.add_parser("provenance")
    provenance.add_argument("--output", required=True)
    provenance.add_argument("--repo-root", required=True)
    provenance.add_argument("--master-seed", type=int, required=True)
    provenance.add_argument("--training-seed", type=int, required=True)
    provenance.add_argument("--evaluation-seed", type=int, required=True)
    provenance.add_argument("--device", required=True)
    provenance.add_argument("--config", action="append", default=[])
    provenance.set_defaults(handler=write_provenance)

    status = subparsers.add_parser("status")
    status.add_argument("--output", required=True)
    status.add_argument("--stage", required=True)
    status.add_argument("--message", default="")
    status.add_argument("--exit-code", type=int, default=None)
    status.add_argument("--last-completed-iteration", type=int, default=None)
    status.set_defaults(handler=write_status)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
