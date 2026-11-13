"""Check complete FedAvg state, independent local BatchNorm, and attacked buffers."""

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.attacks import bfl_round, dba_round, ipm_updates, model_poisoning_round
from meta_sg.defenses import fedavg, fltrust
from meta_sg.federated import model_buffers, model_state, run_federated, train_local_update


def bn_model():
    return nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))


def loader(offset):
    return DataLoader(TensorDataset(torch.tensor([[0., 2.], [2., 4.]]) + offset,
                                    torch.tensor([0, 1])), batch_size=2)


def test_fedavg_matches_independent_full_state_training_across_rounds():
    model = bn_model()
    reference = deepcopy(model)
    clients = [loader(0), loader(10)]
    records = run_federated(model, clients, clients[0], fedavg, rounds=2, clients_per_round=2,
                            local_steps=2, learning_rate=.1, device='cpu', seed=42)
    next(records)
    for round_index in [1, 2]:
        local_states = []
        for client in clients:
            worker = deepcopy(reference)
            train_local_update(worker, model_state(reference), client, steps=2,
                               learning_rate=.1, device='cpu', base_buffers=model_buffers(reference))
            local_states.append(deepcopy(worker.state_dict()))
        expected = {}
        for name, value in local_states[0].items():
            expected[name] = (local_states[0][name] + local_states[1][name]) / 2 if value.is_floating_point() else value
        reference.load_state_dict(expected)
        record = next(records)
        assert record['round'] == round_index
        assert record['aggregation']['buffer_aggregation'] == 'mean_float_max_counter'
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-7)
        assert int(model[0].num_batches_tracked) == round_index * 2


def test_root_training_does_not_replace_client_buffers():
    model = bn_model()
    client = loader(0)
    root = loader(1000)
    records = list(run_federated(model, [client], client, fltrust, rounds=1, clients_per_round=1,
                                 local_steps=1, root_loader=root, root_steps=3,
                                 learning_rate=.1, device='cpu', seed=42))
    torch.testing.assert_close(model[0].running_mean, torch.tensor([.1, .3]))
    assert int(model[0].num_batches_tracked) == 1
    assert records[1]['root_metrics']['steps'] == 3


def test_synthetic_model_poisoning_keeps_global_buffers_for_malicious_submissions():
    model = bn_model()
    clients = [loader(1000), loader(0), loader(10)]
    attack = partial(model_poisoning_round, craft=ipm_updates, malicious_clients=[0])
    list(run_federated(model, clients, clients[1], fedavg, rounds=1, clients_per_round=3,
                       local_steps=1, learning_rate=.1, device='cpu', seed=42, attack=attack))
    torch.testing.assert_close(model[0].running_mean, torch.tensor([1.2 / 3, 1.6 / 3]))
    assert int(model[0].num_batches_tracked) == 1
    assert bool(model[0].running_var.gt(0).all())


@pytest.mark.parametrize('attack_function', [dba_round, bfl_round])
def test_backdoor_buffers_reset_per_client_and_use_max_counter(attack_function):
    model = bn_model()
    clients = [loader(0), loader(10)]

    def trigger(images, *, trigger_index=None):
        return images + 1

    attack = partial(attack_function, malicious_clients=[0], trigger=trigger, target_label=1,
                     malicious_steps=3)
    records = list(run_federated(model, clients, clients[0], fedavg, rounds=1, clients_per_round=2,
                                 local_steps=1, learning_rate=.1, device='cpu', seed=42, attack=attack))
    assert int(model[0].num_batches_tracked) == 3
    assert model[0].num_batches_tracked.dtype == torch.int64
    assert records[1]['attack']['malicious_clients'] == [0]
