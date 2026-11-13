"""Check SGD deltas, worker reset, loader reuse, and sample-weighted loss."""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.federated import model_state, train_local_update


def test_one_step_matches_known_sgd_delta_and_resets_worker_between_clients():
    base_state = (torch.zeros(2, 2), torch.zeros(2))
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.tensor([[1., 0.], [0., 2.]]), torch.tensor([0, 1])), batch_size=2)
    first, metrics = train_local_update(
        model, base_state, loader, steps=1, learning_rate=0.2, device='cpu',
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    second, _ = train_local_update(
        model, base_state, loader, steps=1, learning_rate=0.2, device='cpu',
    )

    torch.testing.assert_close(first[0], torch.tensor([[.05, -.1], [-.05, .1]]))
    torch.testing.assert_close(first[1], torch.zeros(2))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert abs(metrics['loss'] - math.log(2)) < 1e-6
    assert metrics['examples_seen'] == 2 and metrics['steps'] == 1
    assert all(torch.count_nonzero(part) == 0 for part in base_state)
    assert all(not part.requires_grad for part in first)


def test_loader_restart_matches_the_same_explicit_batch_sequence():
    inputs = torch.tensor([[1., 0.], [0., 2.]])
    labels = torch.tensor([0, 1])
    sequence = torch.tensor([0, 1, 0, 1, 0])
    short = DataLoader(TensorDataset(inputs, labels), batch_size=1)
    expanded = DataLoader(TensorDataset(inputs[sequence], labels[sequence]), batch_size=1)
    base_state = model_state(nn.Linear(2, 2))
    restarted, metrics = train_local_update(
        nn.Linear(2, 2), base_state, short, steps=5, learning_rate=0.1, device='cpu',
    )
    explicit, reference_metrics = train_local_update(
        nn.Linear(2, 2), base_state, expanded, steps=5, learning_rate=0.1, device='cpu',
    )

    torch.testing.assert_close(restarted, explicit, rtol=0, atol=0)
    assert metrics == reference_metrics
    assert metrics['examples_seen'] == 5


def test_loss_weights_unequal_batches_by_processed_example_count():
    model = nn.Linear(2, 2)
    base_state = (torch.eye(2), torch.zeros(2))
    inputs = torch.tensor([[0., 2.], [0., 1.], [0., 3.], [4., 0.], [3., 0.]])
    labels = torch.tensor([0, 1, 0, 0, 0])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)
    expected_loss = float(nn.CrossEntropyLoss()(inputs, labels))
    update, metrics = train_local_update(
        model, base_state, loader, steps=2, learning_rate=0, device='cpu',
    )

    assert abs(metrics['loss'] - expected_loss) < 1e-6
    assert metrics['examples_seen'] == 5
    assert all(torch.count_nonzero(part) == 0 for part in update)


def test_zero_steps_restores_global_parameters_and_returns_zero_update():
    base_state = model_state(nn.Linear(2, 2))
    model = nn.Linear(2, 2).eval()
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1, dtype=torch.long)))
    update, metrics = train_local_update(
        model, base_state, loader, steps=0, learning_rate=0.1, device='cpu',
    )

    torch.testing.assert_close(model_state(model), base_state, rtol=0, atol=0)
    assert model.training
    assert all(torch.count_nonzero(part) == 0 for part in update)
    assert metrics == {'loss': None, 'examples_seen': 0, 'steps': 0}


def test_full_epochs_keep_partial_batches_and_match_sgd_optimizer_settings():
    images = torch.tensor([[1., 0.], [0., 2.], [1., 3.], [-1., 2.], [2., 1.]])
    labels = torch.tensor([0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    model = nn.Linear(2, 2)
    base = model_state(model)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=.03, momentum=.8, weight_decay=.02)
    for _ in range(3):
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(batch), targets).backward()
            optimizer.step()
    update, metrics = train_local_update(model, base, loader, steps=99, epochs=3,
                                         learning_rate=.03, momentum=.8, weight_decay=.02, device='cpu')
    torch.testing.assert_close(model_state(model), model_state(reference), rtol=0, atol=0)
    assert metrics['steps'] == 6 and metrics['examples_seen'] == 15
    repeated, _ = train_local_update(model, base, loader, steps=99, epochs=3,
                                     learning_rate=.03, momentum=.8, weight_decay=.02, device='cpu')
    torch.testing.assert_close(update, repeated, rtol=0, atol=0)
