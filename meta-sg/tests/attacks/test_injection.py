"""Check sampled identities, warmup, one-pass training, backdoor schedules, and RL IO."""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from meta_sg.attacks import bfl_round, dba_round, evaluate_asr, ipm_updates, model_poisoning_round, rl_round
from meta_sg.datasets.mnist import dba_trigger
from meta_sg.defenses import fedavg
from meta_sg.federated import model_state, run_federated


class CountedDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels, self.reads = images, labels, 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.reads += 1
        return self.images[index], self.labels[index]


def test_model_poisoning_respects_warmup_and_skips_unused_malicious_training():
    datasets = [CountedDataset(torch.tensor([[1., 0.]]), torch.tensor([i % 2])) for i in range(3)]
    loaders = [DataLoader(dataset) for dataset in datasets]
    evaluation = DataLoader(TensorDataset(torch.ones(1, 2), torch.tensor([0])))
    attack = partial(model_poisoning_round, malicious_clients={2},
                     craft=partial(ipm_updates, scaling_factor=5), attack_start=2)
    records = list(run_federated(nn.Linear(2, 2), loaders, evaluation, fedavg,
                                rounds=2, clients_per_round=3, local_steps=1,
                                learning_rate=.1, device='cpu', seed=42, attack=attack))
    assert records[1]['attack']['malicious_clients'] == []
    assert records[2]['attack']['malicious_clients'] == [2]
    assert [dataset.reads for dataset in datasets] == [2, 2, 1]
    assert {row['client_id'] for row in records[2]['client_metrics']} == {0, 1}


@pytest.mark.parametrize('kind', ['dba', 'bfl'])
def test_backdoors_use_persistent_global_identities_and_train_each_client_once(kind):
    labels = torch.tensor([0, 1, 2, 0])
    datasets = [CountedDataset(torch.zeros(4, 1, 28, 28), labels) for _ in range(5)]
    loaders = [DataLoader(dataset, batch_size=4) for dataset in datasets]
    evaluation = DataLoader(TensorDataset(torch.zeros(1, 1, 28, 28), torch.tensor([0])))
    arguments = dict(malicious_clients={1, 4}, trigger=dba_trigger, target_label=2)
    attack = partial(dba_round, poison_fraction=.25, **arguments) if kind == 'dba' else partial(bfl_round, **arguments)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    backdoor_evaluation = partial(evaluate_asr, loader=evaluation, trigger=dba_trigger,
                                  target_label=2, source_label=None if kind == 'dba' else 0,
                                  device='cpu')
    records = list(run_federated(model, loaders, evaluation, fedavg, rounds=1,
                                clients_per_round=5, local_steps=1, learning_rate=.01,
                                device='cpu', seed=42, attack=attack,
                                backdoor_evaluation=backdoor_evaluation))
    row = records[1]
    assert set(row['attack']['malicious_clients']) == {1, 4}
    assert row['attack']['trigger_assignment'] == ({1: 0, 4: 1} if kind == 'dba' else {1: None, 4: None})
    assert [dataset.reads for dataset in datasets] == [4] * 5
    malicious_rows = [value for value in row['client_metrics'] if value['client_id'] in {1, 4}]
    assert all(value['poisoned_examples'] == (1 if kind == 'dba' else 2) for value in malicious_rows)
    assert row['attack']['model_replacement_scale'] == (10 if kind == 'dba' else 1)
    for record in records:
        assert record['backdoor_evaluation']['examples_seen'] == 1
        assert 0 <= record['backdoor_evaluation']['asr'] <= 1


def test_cyclic_dba_schedule_uses_population_rank_after_warmup():
    loader = DataLoader(TensorDataset(torch.zeros(1, 1, 28, 28), torch.tensor([0])))
    attack = partial(dba_round, malicious_clients={1, 4}, trigger=dba_trigger,
                     target_label=2, attack_start=2, cyclic=True)
    records = list(run_federated(nn.Sequential(nn.Flatten(), nn.Linear(784, 10)),
                                [loader] * 5, loader, fedavg, rounds=3, clients_per_round=5,
                                local_steps=1, learning_rate=.01, device='cpu', seed=42, attack=attack))
    assert records[1]['attack']['malicious_clients'] == []
    assert records[2]['attack']['malicious_clients'] == [1]
    assert records[3]['attack']['malicious_clients'] == [4]


def test_rl_injection_uses_global_observation_and_one_shared_crafted_update():
    model = nn.Linear(128, 10)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    images = torch.zeros(1, 128)
    images[:, 0] = 1
    honest = DataLoader(TensorDataset(images, torch.tensor([0])))
    malicious_data = CountedDataset(images, torch.tensor([1]))
    calls = []

    def actor(state, count):
        calls.append(count)
        assert not any(part.any() for part in state)
        return np.array([-1., -1.], dtype=np.float32), {'pram': np.zeros(1290, dtype=np.float32)}

    attack = partial(rl_round, malicious_clients={1}, actor=actor,
                     known_images=images, known_labels=torch.tensor([0]))
    records = list(run_federated(model, [honest, DataLoader(malicious_data)], honest, fedavg,
                                rounds=1, clients_per_round=2, local_steps=1,
                                learning_rate=.1, device='cpu', seed=42, attack=attack))
    expected = torch.full((10,), -.005)
    expected[0] = .045
    torch.testing.assert_close(model.weight[:, 0], expected)
    torch.testing.assert_close(model.bias, expected)
    assert calls == [1] and malicious_data.reads == 0
    assert records[1]['attack']['gamma'] == 0 and records[1]['attack']['local_steps'] == 0


def test_rl_round_isolates_randomness_and_copies_one_trained_update_for_colluders():
    model = nn.Linear(128, 10)
    base = model_state(model)
    known = torch.linspace(-1, 1, 256).reshape(2, 128)
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(known, labels))
    calls = []

    def actor(state, count):
        calls.append(count)
        return np.array([0., -.88], dtype=np.float32), {'pram': np.zeros(1290)}

    before = torch.get_rng_state()
    updates, metrics, info = rl_round(
        model, base, [loader, loader], [0, 1], round_index=1, steps=1,
        learning_rate=.1, device='cpu', seed=42, malicious_clients={0, 1},
        actor=actor, known_images=known, known_labels=labels,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert calls == [2] and metrics == []
    assert info['local_steps'] == 3 and info['attacker_examples_seen'] == 6
    torch.testing.assert_close(updates[0], updates[1], rtol=0, atol=0)
    saved = updates[1][0].clone()
    updates[0][0].add_(1)
    torch.testing.assert_close(updates[1][0], saved, rtol=0, atol=0)


def test_omniscient_model_poisoning_reports_missing_honest_references():
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.tensor([0])))
    attack = partial(model_poisoning_round, malicious_clients={0}, craft=ipm_updates)
    with pytest.raises(ValueError, match='honest reference'):
        list(run_federated(nn.Linear(2, 2), [loader], loader, fedavg, rounds=1,
                           clients_per_round=1, local_steps=1, learning_rate=.1,
                           device='cpu', seed=42, attack=attack))


@pytest.mark.parametrize('attacker_rate', [.01, .05])
def test_rl_round_keeps_attacker_and_honest_learning_rates_independent(attacker_rate):
    model = nn.Linear(128, 10)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    images = torch.zeros(1, 128)
    images[:, 0] = 1
    loader = DataLoader(TensorDataset(images, torch.tensor([0])))

    def actor(state, count):
        return np.array([0., -.96], dtype=np.float32), {'pram': np.zeros(1290)}

    attack = partial(
        rl_round, malicious_clients={1}, actor=actor,
        known_images=images.repeat(200, 1), known_labels=torch.zeros(200, dtype=torch.long),
        attacker_learning_rate=attacker_rate,
    )
    records = list(run_federated(
        model, [loader, loader], loader, fedavg, rounds=1, clients_per_round=2,
        local_steps=1, learning_rate=.1, device='cpu', seed=42, attack=attack,
    ))
    expected = torch.full((10,), (-.01 + .5 * attacker_rate) / 2)
    expected[0] = (.09 - 4.5 * attacker_rate) / 2
    torch.testing.assert_close(model.weight[:, 0], expected)
    torch.testing.assert_close(model.bias, expected)
    assert records[1]['attack']['attacker_learning_rate'] == attacker_rate


def test_rl_honest_clients_accept_full_epochs_and_sgd_settings():
    images = torch.zeros(3, 128)
    labels = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    def actor(state, count):
        return np.array([-1., -1.], dtype=np.float32), {'pram': np.zeros(1290)}

    attack = partial(rl_round, malicious_clients=[1], actor=actor, known_images=images, known_labels=labels)
    history = list(run_federated(nn.Linear(128, 10), [loader] * 2, loader, fedavg,
                                 rounds=1, clients_per_round=2, local_steps=99, local_epochs=2,
                                 learning_rate=.01, momentum=.8, weight_decay=.001,
                                 device='cpu', seed=42, attack=attack))
    assert history[1]['client_metrics'][0]['steps'] == 4
    assert history[1]['client_metrics'][0]['examples_seen'] == 6
    assert history[1]['attack']['local_steps'] == 0


def test_malicious_step_override_takes_priority_over_honest_epochs():
    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    attack = partial(bfl_round, malicious_clients=[0], trigger=dba_trigger, malicious_steps=1)
    history = list(run_federated(model, [loader] * 2, loader, fedavg, rounds=1, clients_per_round=2,
                                 local_steps=99, local_epochs=2, learning_rate=.01, device='cpu', seed=42,
                                 attack=attack))
    metrics = {row['client_id']: row for row in history[1]['client_metrics']}
    assert metrics[0]['steps'] == 1 and metrics[0]['examples_seen'] == 2
    assert metrics[1]['steps'] == 4 and metrics[1]['examples_seen'] == 8
