"""Check geometric-median objective reduction and relative-tolerance stopping."""

import pytest
import torch

from meta_sg.defenses import rfa


def test_rfa_decreases_geometric_objective_and_moves_toward_majority(updates):
    values = torch.tensor([[0, 0], [.1, .1], [-.1, -.1], [.2, 0], [100, 100]], dtype=torch.float64)
    update, diagnostics = rfa(updates(values))
    before = torch.linalg.vector_norm(values - values.mean(dim=0), dim=1).mean()
    assert diagnostics['rfa_objective'] < float(before)
    assert update[0].norm() < 1
    assert diagnostics['rfa_iterations'] == 4


def test_rfa_relative_tolerance_stops_early(updates):
    _, diagnostics = rfa(updates([[0, 0], [.1, .1], [100, 100]]), iterations=100, ftol=1)
    assert diagnostics['rfa_iterations'] == 1


@pytest.mark.parametrize('values', [[[0, 0]] * 3, [[2, -3]]])
def test_rfa_stops_when_the_geometric_objective_is_zero(updates, values):
    clients = updates(values)
    update, diagnostics = rfa(clients, iterations=100)

    torch.testing.assert_close(update, clients[0], rtol=0, atol=0)
    assert diagnostics['rfa_objective'] == 0
    assert diagnostics['rfa_iterations'] == 1
