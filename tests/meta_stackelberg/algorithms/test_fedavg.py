import numpy as np
import pytest

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.aggregation.fedavg import FedAvg
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.types import ClientUpdate


def _update(client_id: int, values: list[float], num_examples: int) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        delta=ModelState.from_tensors([np.asarray(values, dtype=np.float32)]),
        num_examples=num_examples,
    )


def test_fedavg_weights_client_deltas_by_number_of_examples() -> None:
    aggregate = FedAvg().aggregate([
        _update(0, [1.0, 1.0], num_examples=1),
        _update(1, [3.0, 5.0], num_examples=3),
    ])

    np.testing.assert_allclose(aggregate.vector(), [2.5, 4.0])


def test_fedavg_reduces_to_arithmetic_mean_for_equal_counts() -> None:
    aggregate = FedAvg().aggregate([
        _update(0, [1.0, 3.0], num_examples=2),
        _update(1, [5.0, 7.0], num_examples=2),
    ])

    np.testing.assert_allclose(aggregate.vector(), [3.0, 5.0])


def test_fedavg_rejects_empty_or_incompatible_updates() -> None:
    with pytest.raises(ValueError, match='at least one'):
        FedAvg().aggregate([])
    with pytest.raises(ValueError, match='structure'):
        FedAvg().aggregate([
            _update(0, [1.0, 2.0], num_examples=1),
            _update(1, [3.0], num_examples=1),
        ])


def test_server_sgd_applies_aggregate_delta_with_learning_rate() -> None:
    model = ModelState.from_tensors([np.asarray([10.0, 20.0], dtype=np.float32)])
    delta = ModelState.from_tensors([np.asarray([2.0, -4.0], dtype=np.float32)])

    updated = ServerSGD().step(model, delta, learning_rate=0.5)

    np.testing.assert_allclose(updated.vector(), [11.0, 18.0])


@pytest.mark.parametrize('learning_rate', [-0.1, float('nan'), float('inf')])
def test_server_sgd_rejects_invalid_learning_rates(learning_rate: float) -> None:
    model = ModelState.from_tensors([np.asarray([1.0], dtype=np.float32)])
    delta = ModelState.from_tensors([np.asarray([1.0], dtype=np.float32)])

    with pytest.raises(ValueError, match='learning_rate'):
        ServerSGD().step(model, delta, learning_rate=learning_rate)
