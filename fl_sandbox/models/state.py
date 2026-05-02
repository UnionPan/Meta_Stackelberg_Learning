"""State compression helpers used by adaptive attackers."""

from __future__ import annotations

try:
    from src.models.cnn import get_compressed_state as _get_compressed_state
except ImportError:  # pragma: no cover
    _get_compressed_state = None


def get_compressed_state(model, num_tail_layers: int = 2):
    """Return the compressed model state expected by RL attackers."""
    if _get_compressed_state is None:
        raise RuntimeError("get_compressed_state is unavailable; check src.models.cnn")
    return _get_compressed_state(model, num_tail_layers=num_tail_layers)
