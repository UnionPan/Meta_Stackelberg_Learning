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

    def parameter_tensor_count(self, model: torch.nn.Module) -> int:
        return len(tuple(model.parameters()))


class TorchModelStateCodec(TorchParameterCodec):
    """Capture parameters plus floating persistent buffers such as BatchNorm stats.

    Integer counters are excluded because ModelState is a floating, aggregatable
    FL state. Standard BatchNorm momentum does not depend on num_batches_tracked.
    """

    def capture(self, model: torch.nn.Module) -> ModelState:
        values = tuple(model.parameters()) + _floating_buffers(model)
        return ModelState.from_tensors(
            value.detach().cpu().numpy().copy() for value in values
        )

    def load(self, model: torch.nn.Module, state: ModelState) -> None:
        values = tuple(model.parameters()) + _floating_buffers(model)
        if len(values) != len(state.tensors):
            raise ValueError(
                f'model-state count {len(values)} does not match state count {len(state.tensors)}'
            )
        with torch.no_grad():
            for index, (value, tensor) in enumerate(zip(values, state.tensors)):
                if tuple(value.shape) != tensor.shape:
                    raise ValueError(
                        f'model-state shape mismatch at index {index}: '
                        f'{tuple(value.shape)} != {tensor.shape}'
                    )
                value.copy_(
                    torch.tensor(
                        tensor, device=value.device, dtype=value.dtype,
                    )
                )


def _floating_buffers(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(
        buffer for buffer in model.buffers() if torch.is_floating_point(buffer)
    )
