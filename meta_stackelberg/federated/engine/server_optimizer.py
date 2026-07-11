"""Server-side application of aggregate model deltas."""

from __future__ import annotations

import math

from meta_stackelberg.core.model_state import ModelState, apply_delta


class ServerSGD:
    """Apply ``model + learning_rate * aggregate_delta``."""

    def step(
        self,
        model: ModelState,
        aggregate_delta: ModelState,
        learning_rate: float,
    ) -> ModelState:
        value = float(learning_rate)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError('learning_rate must be finite and non-negative')
        return apply_delta(model, aggregate_delta, scale=value)
