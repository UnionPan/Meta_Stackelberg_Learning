"""Model weight conversion helpers used by attacks and aggregators."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - pure numpy helpers still work
    torch = None


def weights_to_vector(weights: Iterable[np.ndarray]) -> np.ndarray:
    """Convert a sequence of layer arrays into one flat vector."""
    return np.concatenate([np.asarray(weight).ravel() for weight in weights], axis=0)


def vector_to_weights(vector: np.ndarray, weights_template: list[np.ndarray]) -> list[np.ndarray]:
    """Reshape a flat vector to match a list-of-arrays weight template."""
    boundaries = np.cumsum([0] + [weight.size for weight in weights_template])
    return [
        np.asarray(vector[boundaries[idx] : boundaries[idx + 1]]).reshape(weights_template[idx].shape)
        for idx in range(len(weights_template))
    ]


def get_parameters(model) -> list[np.ndarray]:
    """Capture a torch model state dict as numpy arrays."""
    if torch is None:
        raise RuntimeError("torch is required to capture model parameters")
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_parameters(model, parameters: list[np.ndarray]) -> None:
    """Load numpy arrays into a torch model state dict."""
    if torch is None:
        raise RuntimeError("torch is required to set model parameters")
    model_state = model.state_dict()
    state_dict = OrderedDict(
        (key, torch.as_tensor(value, device=model_state[key].device, dtype=model_state[key].dtype))
        for key, value in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state_dict, strict=True)
