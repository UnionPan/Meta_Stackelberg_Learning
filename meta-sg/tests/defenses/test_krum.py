"""Check Krum selection against scalar distances, tie order, and client-count requirements."""

import pytest
import torch

from meta_sg.defenses import krum, multi_krum


@pytest.mark.parametrize('multiple', [False, True])
def test_krum_matches_independent_scalar_distance_reference(updates, multiple):
    values = [[0, 0], [.1, .1], [-.1, 0], [0, .2], [.2, -.1], [30, 40], [-40, -30]]
    distances = [
        [sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
         for j, right in enumerate(values) if j != i]
        for i, left in enumerate(values)
    ]
    scores = [sum(sorted(row)[:4]) for row in distances]
    selected = sorted(range(7), key=lambda index: scores[index])[:3 if multiple else 1]
    if multiple:
        update, diagnostics = multi_krum(updates(values), byzantine_bound=1, selected=3)
    else:
        update, diagnostics = krum(updates(values), byzantine_bound=1)
    expected = torch.tensor([values[i] for i in selected], dtype=torch.float64).mean(dim=0)
    assert diagnostics['selected_indices'] == selected
    torch.testing.assert_close(update[0], expected)


@pytest.mark.parametrize('defense', [krum, multi_krum])
def test_krum_rejects_insufficient_clients(updates, defense):
    with pytest.raises(ValueError):
        defense(updates([[1]] * 7))


def test_multikrum_rejects_excessive_selection(updates):
    with pytest.raises(ValueError, match='selected <= n'):
        multi_krum(updates([[1]] * 7), byzantine_bound=1, selected=8)


def test_equal_scores_select_first_clients_stably(updates):
    clients = updates([[2, -3]] * 7)
    _, single = krum(clients, byzantine_bound=1)
    _, multiple = multi_krum(clients, byzantine_bound=1, selected=3)
    assert single['selected_indices'] == [0]
    assert multiple['selected_indices'] == [0, 1, 2]
