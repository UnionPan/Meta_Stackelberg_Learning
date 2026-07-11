"""Model-aware NeuroClip on an independent post-training model copy."""

from __future__ import annotations

import copy
import math

import torch


class NeuroClipCopy(torch.nn.Module):
    """Clone a CNN and clamp positive convolution activations to epsilon."""

    def __init__(self, model: torch.nn.Module, epsilon: float) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise TypeError('model must be a torch.nn.Module')
        if isinstance(epsilon, bool):
            raise TypeError('epsilon must be a real number')
        checked = float(epsilon)
        if not math.isfinite(checked) or checked <= 0.0:
            raise ValueError('epsilon must be finite and positive')
        self.epsilon = checked
        self.model = copy.deepcopy(model)
        self._hook_handles = tuple(
            module.register_forward_hook(self._clamp_activation)
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        if not self._hook_handles:
            raise ValueError('NeuroClip requires at least one convolutional layer')

    def _clamp_activation(self, module, inputs, output):
        del module, inputs
        return torch.clamp(output, max=self.epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
