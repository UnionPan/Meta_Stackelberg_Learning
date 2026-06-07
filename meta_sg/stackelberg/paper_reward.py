"""Paper-aligned defender reward for Stackelberg FL experiments."""

from __future__ import annotations

import numpy as np


def compute_paper_defender_reward(info: dict, *, scale: float = 1.0) -> float:
    if "post_clean_loss" not in info:
        raise KeyError("post_clean_loss is required for paper reward r_D = -F(h(w))")
    return -float(scale) * _finite(info["post_clean_loss"])


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if np.isfinite(result) else float(default)
