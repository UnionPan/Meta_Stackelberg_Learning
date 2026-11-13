"""Check weighted model updates and model preservation when all clients are rejected."""

import pytest
import torch

from meta_sg.federated.aggregation import aggregate_client_updates


def test_weighted_update_aggregation() -> None:
    current = (torch.tensor([1.0, 2.0]),)
    result = aggregate_client_updates(
        current,
        [(torch.tensor([1.0, 0.0]),), (torch.tensor([3.0, 2.0]),)],
        weights=[1.0, 3.0],
    )
    assert torch.allclose(result[0], torch.tensor([3.5, 3.5]))


def test_rejecting_all_clients_preserves_global_model() -> None:
    current = (torch.tensor([7.]),)
    result = aggregate_client_updates(
        current, [(torch.tensor([100.]),)] * 3, weights=[0., 0., 0.],
    )
    assert torch.equal(result[0], current[0])


@pytest.mark.parametrize('scale', [1e-13, 1e-25])
def test_positive_weight_rescaling_preserves_the_aggregated_update(scale):
    current = (torch.tensor([1., 2.]),)
    updates = [(torch.tensor([1., 0.]),), (torch.tensor([3., 2.]),)]
    expected = aggregate_client_updates(current, updates, weights=[1., 3.])
    actual = aggregate_client_updates(current, updates, weights=[scale, 3 * scale])

    torch.testing.assert_close(actual, expected)
