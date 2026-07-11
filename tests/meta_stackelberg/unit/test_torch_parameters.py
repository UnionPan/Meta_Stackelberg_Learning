import numpy as np
import pytest
import torch

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.models.parameters import (
    TorchModelStateCodec,
    TorchParameterCodec,
)


def _linear() -> torch.nn.Linear:
    return torch.nn.Linear(2, 2, bias=True)


def test_parameter_codec_round_trip_preserves_values_shapes_and_dtypes() -> None:
    source = _linear()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        source.bias.copy_(torch.tensor([5.0, 6.0]))
    codec = TorchParameterCodec()
    state = codec.capture(source)
    target = _linear()

    codec.load(target, state)

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)
    assert [tensor.shape for tensor in state.tensors] == [(2, 2), (2,)]
    assert all(tensor.dtype == np.float32 for tensor in state.tensors)


def test_captured_state_does_not_share_storage_with_model() -> None:
    model = _linear()
    codec = TorchParameterCodec()
    state = codec.capture(model)
    captured = state.vector().copy()

    with torch.no_grad():
        model.weight.add_(100.0)

    np.testing.assert_array_equal(state.vector(), captured)


def test_parameter_codec_rejects_count_and_shape_mismatches() -> None:
    model = _linear()
    codec = TorchParameterCodec()
    with pytest.raises(ValueError, match='count'):
        codec.load(
            model,
            ModelState.from_tensors([np.zeros((2, 2), dtype=np.float32)]),
        )
    with pytest.raises(ValueError, match='shape'):
        codec.load(
            model,
            ModelState.from_tensors([
                np.zeros((4,), dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
            ]),
        )


def test_full_model_codec_round_trips_batchnorm_float_buffers_but_not_counter() -> None:
    source = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    source[1].running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    source[1].running_var.copy_(torch.tensor([4.0, 5.0, 6.0]))
    source[1].num_batches_tracked.fill_(7)
    codec = TorchModelStateCodec()

    state = codec.capture(source)
    target = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    codec.load(target, state)

    assert codec.parameter_tensor_count(source) == 4
    assert len(state.tensors) == 6
    torch.testing.assert_close(target[1].running_mean, source[1].running_mean)
    torch.testing.assert_close(target[1].running_var, source[1].running_var)
    assert target[1].num_batches_tracked.item() == 0


def test_full_model_codec_rejects_buffer_shape_mismatch() -> None:
    codec = TorchModelStateCodec()
    model = torch.nn.BatchNorm1d(3)
    state = codec.capture(model)
    with pytest.raises(ValueError, match='shape'):
        codec.load(
            model,
            ModelState.from_tensors(
                list(state.tensors[:-1]) + [np.zeros(4, dtype=np.float32)]
            ),
        )
