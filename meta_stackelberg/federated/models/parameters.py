"""Conversion between PyTorch trainable parameters and ModelState."""

from __future__ import annotations

import torch

from meta_stackelberg.core.model_state import ModelState


class TorchParameterCodec:
    """Capture and load trainable parameters in module iteration order."""

    def capture(self, model: torch.nn.Module) -> ModelState:
        return ModelState.from_tensors(
            parameter.detach().cpu().numpy().copy()
            for parameter in model.parameters()
        )

    def load(self, model: torch.nn.Module, state: ModelState) -> None:
        parameters = tuple(model.parameters())
        if len(parameters) != len(state.tensors):
            raise ValueError(
                f'parameter count {len(parameters)} does not match state count {len(state.tensors)}'
            )
        with torch.no_grad():
            for index, (parameter, tensor) in enumerate(zip(parameters, state.tensors)):
                if tuple(parameter.shape) != tensor.shape:
                    raise ValueError(
                        f'parameter shape mismatch at index {index}: '
                        f'{tuple(parameter.shape)} != {tensor.shape}'
                    )
                parameter.copy_(
                    torch.tensor(tensor, device=parameter.device, dtype=parameter.dtype)
                )
