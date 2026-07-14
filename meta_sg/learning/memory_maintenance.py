"""Best-effort task-boundary host-memory maintenance and telemetry."""
from __future__ import annotations

import ctypes
import gc
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryMaintenanceResult:
    """Bounded, JSON-safe outcome of one memory-maintenance pass."""

    elapsed_seconds: float
    objects_collected: int
    rss_before_kib: int | None
    rss_after_kib: int | None
    rss_released_kib: int | None
    malloc_trim_supported: bool
    malloc_trim_succeeded: bool
    warning: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _bounded_warning(value: object) -> str:
    return str(value).replace("\n", " ")[:240]


def _read_rss_kib() -> int | None:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _malloc_trim() -> tuple[bool, bool, str | None]:
    if platform.system() != "Linux":
        return False, False, "malloc_trim unsupported on this platform"
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return True, bool(trim(0)), None
    except (AttributeError, OSError, TypeError) as exc:
        return False, False, _bounded_warning(exc)


def perform_memory_maintenance() -> MemoryMaintenanceResult:
    """Collect unreachable objects and return freed pages to glibc when possible."""

    started = time.perf_counter()
    rss_before = _read_rss_kib()
    objects_collected = int(gc.collect())
    trim_supported, trim_succeeded, warning = _malloc_trim()
    rss_after = _read_rss_kib()
    rss_released = (
        None
        if rss_before is None or rss_after is None
        else max(0, rss_before - rss_after)
    )
    return MemoryMaintenanceResult(
        elapsed_seconds=time.perf_counter() - started,
        objects_collected=objects_collected,
        rss_before_kib=rss_before,
        rss_after_kib=rss_after,
        rss_released_kib=rss_released,
        malloc_trim_supported=trim_supported,
        malloc_trim_succeeded=trim_succeeded,
        warning=warning,
    )
