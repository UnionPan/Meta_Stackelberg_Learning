import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.protocols import Aggregator
from meta_stackelberg.federated.types import ClientUpdate


def _update(client_id: int, first, second) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors((
            np.asarray(first, dtype=np.float32),
            np.asarray(second, dtype=np.float64),
        )),
        num_examples=client_id + 1,
    )


def test_coordinate_median_aggregates_each_coordinate_and_preserves_dtype() -> None:
    from meta_stackelberg.federated.aggregation.coordinate_median import CoordinateMedian

    aggregator = CoordinateMedian()
    result = aggregator.aggregate((
        _update(0, [1.0, 100.0], [[3.0]]),
        _update(1, [2.0, 4.0], [[1.0]]),
        _update(2, [8.0, 5.0], [[2.0]]),
    ))

    assert isinstance(aggregator, Aggregator)
    np.testing.assert_array_equal(result.tensors[0], [2.0, 5.0])
    np.testing.assert_array_equal(result.tensors[1], [[2.0]])
    assert result.tensors[0].dtype == np.float32
    assert result.tensors[1].dtype == np.float64


def test_coordinate_median_rejects_empty_or_mismatched_updates() -> None:
    from meta_stackelberg.federated.aggregation.coordinate_median import CoordinateMedian

    aggregator = CoordinateMedian()
    with pytest.raises(ValueError, match='at least one'):
        aggregator.aggregate(())
    with pytest.raises(ValueError, match='structure'):
        aggregator.aggregate((
            _update(0, [1.0, 2.0], [[3.0]]),
            ClientUpdate(
                client_id=2,
                delta=ModelState.from_tensors((np.zeros(3, dtype=np.float32),)),
                num_examples=1,
            ),
        ))
