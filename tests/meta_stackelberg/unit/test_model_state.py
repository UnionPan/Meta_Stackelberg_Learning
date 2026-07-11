import numpy as np
import pytest

from meta_stackelberg.core.model_state import (
    ModelState,
    apply_delta,
    from_vector,
    model_difference,
    state_cosine,
    state_l2_norm,
)


def _assert_state(state: ModelState, expected: list[list[float]]) -> None:
    assert len(state.tensors) == len(expected)
    for actual, values in zip(state.tensors, expected):
        np.testing.assert_allclose(actual, np.asarray(values, dtype=actual.dtype))


def test_difference_and_apply_delta_use_local_minus_global_semantics() -> None:
    old = ModelState.from_tensors([
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    ])
    new = ModelState.from_tensors([
        np.asarray([2.0, 4.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    ])

    delta = model_difference(new, old)

    _assert_state(delta, [[1.0, 2.0], [-2.0]])
    _assert_state(apply_delta(old, delta), [[2.0, 4.0], [1.0]])
    _assert_state(apply_delta(old, delta, scale=0.5), [[1.5, 3.0], [2.0]])


def test_model_state_copies_inputs_and_exposes_read_only_tensors() -> None:
    source = np.asarray([1.0, 2.0], dtype=np.float32)
    state = ModelState.from_tensors([source])
    source[0] = 99.0

    np.testing.assert_allclose(state.tensors[0], [1.0, 2.0])
    assert state.tensors[0].flags.writeable is False
    with pytest.raises(ValueError):
        state.tensors[0][0] = 5.0


def test_clone_has_independent_storage() -> None:
    state = ModelState.from_tensors([np.asarray([1.0, 2.0], dtype=np.float32)])
    clone = state.clone()

    assert not np.shares_memory(state.tensors[0], clone.tensors[0])
    np.testing.assert_array_equal(state.tensors[0], clone.tensors[0])


def test_vector_round_trip_preserves_shapes_and_dtypes() -> None:
    template = ModelState.from_tensors([
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
    ])
    vector = np.arange(5, dtype=np.float32)

    restored = from_vector(vector, template)

    np.testing.assert_array_equal(restored.vector(), vector)
    assert [tensor.shape for tensor in restored.tensors] == [(2, 2), (1,)]
    assert all(tensor.dtype == np.float32 for tensor in restored.tensors)


def test_model_state_rejects_empty_and_non_floating_tensors() -> None:
    with pytest.raises(ValueError, match='at least one tensor'):
        ModelState.from_tensors([])
    with pytest.raises(TypeError, match='floating'):
        ModelState.from_tensors([np.asarray([1, 2], dtype=np.int64)])


def test_binary_operations_reject_incompatible_structures() -> None:
    one = ModelState.from_tensors([np.ones(2, dtype=np.float32)])
    two = ModelState.from_tensors([np.ones(3, dtype=np.float32)])

    with pytest.raises(ValueError, match='shape'):
        model_difference(one, two)
    with pytest.raises(ValueError, match='length'):
        from_vector(np.ones(3, dtype=np.float32), one)


def test_norm_and_cosine_are_computed_on_flattened_state() -> None:
    left = ModelState.from_tensors([np.asarray([3.0, 4.0], dtype=np.float32)])
    same_direction = ModelState.from_tensors([np.asarray([6.0, 8.0], dtype=np.float32)])
    orthogonal = ModelState.from_tensors([np.asarray([-4.0, 3.0], dtype=np.float32)])
    zero = ModelState.from_tensors([np.asarray([0.0, 0.0], dtype=np.float32)])

    assert state_l2_norm(left) == pytest.approx(5.0)
    assert state_cosine(left, same_direction) == pytest.approx(1.0)
    assert state_cosine(left, orthogonal) == pytest.approx(0.0, abs=1e-7)
    assert state_cosine(left, zero) == 0.0
