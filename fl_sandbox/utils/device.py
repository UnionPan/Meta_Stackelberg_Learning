"""Runtime device selection helpers."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - lightweight import support
    torch = None


def _default_device():
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device=None):
    """Resolve a device string, ``torch.device``, or ``None``."""
    if torch is None:
        raise RuntimeError("torch is required to resolve FL runtime devices")
    if device is None:
        return _default_device()
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized == "auto":
            return _default_device()
        return torch.device(device)
    raise TypeError(f"Unsupported device spec: {device!r}")
