import math

import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.protocols import Aggregator
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.defenses.trimmed_mean import CoordinateTrimmedMean


def _update(
    client_id: int,
    tensors,
    *,
    examples: int = 1,
    dtype=np.float32,
    metadata=None,
) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors(np.asarray(value, dtype=dtype) for value in tensors),
        num_examples=examples,
        metadata={} if metadata is None else metadata,
    )


def test_trims_each_coordinate_independently_and_ignores_sample_weights() -> None:
    updates = (
        _update(0, [np.array([0.0, 100.0])], examples=1),
        _update(1, [np.array([1.0, 3.0])], examples=100),
        _update(2, [np.array([2.0, 2.0])], examples=1),
        _update(3, [np.array([3.0, 1.0])], examples=1),
        _update(4, [np.array([100.0, 0.0])], examples=1),
    )

    result = CoordinateTrimmedMean(0.2).aggregate(updates)

    np.testing.assert_array_equal(result.vector(), np.array([2.0, 2.0], dtype=np.float32))


def test_zero_trim_is_exact_equal_client_mean_not_weighted_fedavg() -> None:
    updates = (_update(0, [[0.0]], examples=1), _update(1, [[2.0]], examples=99))

    result = CoordinateTrimmedMean(0.0).aggregate(updates)

    np.testing.assert_array_equal(result.vector(), np.array([1.0], dtype=np.float32))


@pytest.mark.parametrize(
    ('client_count', 'ratio', 'trim_count', 'retained'),
    [(4, 0.24, 0, 4), (4, 0.25, 1, 2), (5, 0.2, 1, 3), (6, 0.4, 2, 2)],
)
def test_summary_uses_floor_per_tail(client_count, ratio, trim_count, retained) -> None:
    summary = CoordinateTrimmedMean(ratio).summarize(client_count)

    assert summary.client_count == client_count
    assert summary.trim_ratio == ratio
    assert summary.per_tail_trim_count == trim_count
    assert summary.retained_count == retained


def test_multi_tensor_result_preserves_template_dtype() -> None:
    updates = tuple(
        _update(index, [[value], [value + 1.0]], dtype=np.float64)
        for index, value in enumerate((0.0, 1.0, 2.0))
    )

    result = CoordinateTrimmedMean(0.34).aggregate(updates)

    np.testing.assert_array_equal(result.vector(), np.array([1.0, 2.0]))
    assert all(tensor.dtype == np.float64 for tensor in result.tensors)


def test_permutation_metadata_and_client_ids_do_not_change_result() -> None:
    updates = (
        _update(10, [[0.0]], metadata={'role': 'malicious'}),
        _update(2, [[2.0]], metadata={'role': 'benign'}),
        _update(99, [[100.0]], metadata={'role': 'unknown'}),
    )
    aggregator = CoordinateTrimmedMean(0.34)

    forward = aggregator.aggregate(updates)
    reverse = aggregator.aggregate(tuple(reversed(updates)))

    np.testing.assert_array_equal(forward.vector(), np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(forward.vector(), reverse.vector())


def test_does_not_mutate_input_arrays() -> None:
    updates = (_update(0, [[0.0, 3.0]]), _update(1, [[2.0, 1.0]]))
    before = tuple(update.delta.vector() for update in updates)

    CoordinateTrimmedMean(0.0).aggregate(updates)

    for update, expected in zip(updates, before):
        np.testing.assert_array_equal(update.delta.vector(), expected)


@pytest.mark.parametrize('ratio', [-0.1, 0.5, math.nan, math.inf, True])
def test_rejects_invalid_trim_ratio(ratio: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoordinateTrimmedMean(ratio)  # type: ignore[arg-type]


def test_rejects_empty_non_finite_and_incompatible_updates() -> None:
    aggregator = CoordinateTrimmedMean(0.2)
    with pytest.raises(ValueError, match='at least one'):
        aggregator.aggregate(())
    with pytest.raises(ValueError, match='finite'):
        aggregator.aggregate((_update(0, [[np.nan]]),))
    with pytest.raises(ValueError, match='structure'):
        aggregator.aggregate((_update(0, [[1.0]]), _update(1, [[1.0, 2.0]])))
    with pytest.raises(ValueError, match='dtype'):
        aggregator.aggregate((
            _update(0, [[1.0]], dtype=np.float32),
            _update(1, [[1.0]], dtype=np.float64),
        ))


def test_summary_rejects_invalid_client_count() -> None:
    for count in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            CoordinateTrimmedMean(0.2).summarize(count)  # type: ignore[arg-type]


def test_satisfies_existing_aggregator_protocol() -> None:
    assert isinstance(CoordinateTrimmedMean(0.2), Aggregator)
