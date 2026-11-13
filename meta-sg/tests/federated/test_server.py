"""Check round coordination, participant sampling, aggregation, and trusted-root use."""

from functools import partial
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.defenses import fedavg, fltrust, trimmed_mean
from meta_sg.federated import model_state, run_federated


def zero_model():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


@pytest.mark.parametrize('aggregate', [fedavg, partial(trimmed_mean, byzantine_bound=0)],
                         ids=['fedavg', 'trimmed_mean'])
def test_round_aggregates_independent_client_deltas_from_one_global_snapshot(aggregate):
    images = torch.tensor([[1., 0.], [0., 2.]])
    labels = torch.tensor([0, 1])
    clients = [DataLoader(TensorDataset(images[i:i + 1], labels[i:i + 1])) for i in range(2)]
    test_loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = zero_model()
    history = list(run_federated(
        model, clients, test_loader, aggregate, rounds=1, clients_per_round=2,
        local_steps=1, learning_rate=0.2, device='cpu', seed=42,
    ))

    torch.testing.assert_close(model.weight, torch.tensor([[.05, -.1], [-.05, .1]]))
    torch.testing.assert_close(model.bias, torch.zeros(2))
    assert [row['round'] for row in history] == [0, 1]
    assert abs(history[0]['evaluation']['loss'] - math.log(2)) < 1e-6
    assert history[0]['selected_clients'] == []
    assert set(history[1]['selected_clients']) == {0, 1}
    assert history[1]['evaluation']['accuracy'] == 1
    assert all(client['examples_seen'] == 1 for client in history[1]['client_metrics'])


def test_multiround_participation_is_reproducible_without_duplicates_or_global_rng_changes():
    clients = [
        DataLoader(TensorDataset(torch.tensor([[float(i + 1), 1.]]), torch.tensor([i % 2])))
        for i in range(5)
    ]
    test_loader = DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 1])))
    first_model, second_model = zero_model(), zero_model()
    rng_before = np.random.get_state()
    arguments = dict(rounds=3, clients_per_round=2, local_steps=1,
                     learning_rate=0.1, device='cpu', seed=42)
    first = list(run_federated(first_model, clients, test_loader, fedavg, **arguments))
    second = list(run_federated(second_model, clients, test_loader, fedavg, **arguments))
    rng_after = np.random.get_state()

    assert first == second
    torch.testing.assert_close(model_state(first_model), model_state(second_model), rtol=0, atol=0)
    assert [row['round'] for row in first] == [0, 1, 2, 3]
    for row in first[1:]:
        assert len(set(row['selected_clients'])) == 2
        assert [client['client_id'] for client in row['client_metrics']] == row['selected_clients']
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])


def test_fltrust_trains_root_from_global_parameters_and_applies_root_scale_once():
    loader = DataLoader(TensorDataset(torch.tensor([[1., 0.]]), torch.tensor([0])))
    model = zero_model()
    history = list(run_federated(
        model, [loader], loader, fltrust, rounds=1, clients_per_round=1,
        local_steps=2, learning_rate=0.2, device='cpu', seed=42,
        root_loader=loader, root_steps=1,
    ))

    torch.testing.assert_close(model.weight, torch.tensor([[.1, 0.], [-.1, 0.]]))
    torch.testing.assert_close(model.bias, torch.tensor([.1, -.1]))
    assert history[1]['root_metrics']['steps'] == 1
    assert history[1]['client_metrics'][0]['steps'] == 2
    assert not history[1]['aggregation']['zero_trust']


def test_fltrust_zero_trust_preserves_the_global_model():
    images = torch.tensor([[1., 0.]])
    client = DataLoader(TensorDataset(images, torch.tensor([1])))
    root = DataLoader(TensorDataset(images, torch.tensor([0])))
    model = zero_model()
    history = list(run_federated(
        model, [client], root, fltrust, rounds=1, clients_per_round=1,
        local_steps=1, learning_rate=0.2, device='cpu', seed=42,
        root_loader=root, root_steps=1,
    ))

    assert all(torch.count_nonzero(part) == 0 for part in model_state(model))
    assert history[1]['aggregation']['zero_trust']


def test_round_decay_matches_explicit_sgd_and_records_the_actual_learning_rate():
    model = zero_model()
    reference = deepcopy(model)
    images, labels = torch.eye(2), torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    history = run_federated(model, [loader], loader, fedavg, rounds=3, clients_per_round=1,
                            local_steps=1, learning_rate=.2, learning_rate_decay=.5,
                            momentum=.8, weight_decay=.02, device='cpu', seed=42)
    assert next(history)['learning_rate'] == .2
    for rate in [.2, .1, .05]:
        optimizer = torch.optim.SGD(reference.parameters(), lr=rate, momentum=.8, weight_decay=.02)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(reference(images), labels).backward()
        optimizer.step()
        record = next(history)
        assert record['learning_rate'] == rate
        torch.testing.assert_close(model_state(model), model_state(reference), rtol=1e-6, atol=1e-7)
