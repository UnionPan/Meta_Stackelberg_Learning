"""Exercise JSON-to-round assembly using tiny local datasets, without real jobs."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from meta_sg.experiments.config import DEFAULTS, load_config
from meta_sg.experiments import run
from meta_sg.federated import model_state


def small_config(root, *, dataset='mnist'):
    config = deepcopy(DEFAULTS)
    config['dataset'].update(name=dataset, data_root=str(root))
    config['partition']['root_samples_per_class'] = 1
    config['federation'].update(num_clients=3, clients_per_round=3)
    config['training'].update(rounds=1, local_steps=1, root_steps=1, batch_size=2, learning_rate=.023)
    return config


def test_defaults_are_merged_without_mutating_later_loads(tmp_path):
    path = tmp_path / 'run.json'
    path.write_text(json.dumps({'training': {'learning_rate': .01}, 'defense': {'name': 'median'}}))
    first = load_config(path)
    assert first['training']['learning_rate'] == .01
    assert first['training']['rounds'] == 5
    assert first['defense']['name'] == 'median'
    first['dataset']['options']['download'] = True
    assert load_config(path)['dataset']['options'] == {}
    path.write_text('{"training": {"learn_rate": 0.01}}')
    with pytest.raises(ValueError, match='learn_rate'):
        load_config(path)


@pytest.mark.parametrize('example', ['mnist', 'cifar10', 'tiny_imagenet', 'cifar10_resnet18',
                                    'tiny_imagenet_resnet18', 'tiny_imagenet_bfl', 'mnist_rl', 'mnist_rl_histogram40'])
def test_example_cli_dry_run_does_not_load_data(example, monkeypatch, capsys):
    path = Path(__file__).parents[2] / 'configs' / f'{example}.json'
    config = load_config(path)
    dataset = config['dataset']['name']
    _, trigger = run.DATASETS[dataset]
    model = run.MODELS[dataset][config['model']['name']]

    def no_data(*args, **kwargs):
        pytest.fail('dry-run must not load data')

    monkeypatch.setitem(run.DATASETS, dataset, (no_data, trigger))
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--config', str(path), '--dry-run'])
    run.main()
    output = json.loads(capsys.readouterr().out)
    assert output['config']['dataset']['name'] == dataset
    assert output['resolved']['model'] == model.__name__


@pytest.mark.parametrize('dataset,side,defense,attack', [
    ('cifar10', 32, 'fedavg', 'none'),
    ('tiny_imagenet', 64, 'fltrust', 'dba'),
])
def test_resnet_json_runs_one_synthetic_round(dataset, side, defense, attack, tmp_path, monkeypatch):
    from torch.utils.data import TensorDataset

    def load_fixture(root, **options):
        generator = torch.Generator().manual_seed(5)
        train = TensorDataset(torch.randn(8, 3, side, side, generator=generator), torch.arange(8) % 4)
        test = TensorDataset(torch.randn(4, 3, side, side, generator=generator), torch.arange(4))
        train.targets = train.tensors[1]
        train.transform = test.transform = None
        return train, test

    _, trigger = run.DATASETS[dataset]
    monkeypatch.setitem(run.DATASETS, dataset, (load_fixture, trigger))
    config = small_config(tmp_path, dataset=dataset)
    config['model']['name'] = 'resnet18'
    config['federation'].update(num_clients=2, clients_per_round=2)
    config['defense']['name'] = defense
    config['attack'].update(name=attack, malicious_clients=[0], start_round=1)
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(config))
    model, arguments = run.build_experiment(load_config(path))
    before = model.bn1.running_mean.clone()
    records = list(run.run_federated(model, **arguments))
    assert records[1]['aggregation']['buffer_aggregation'] == 'mean_float_max_counter'
    assert records[1]['evaluation']['examples_seen'] == 4
    assert torch.isfinite(torch.tensor(records[1]['evaluation']['loss']))
    assert int(model.bn1.num_batches_tracked) == 1
    assert not torch.equal(model.bn1.running_mean, before)
    if attack == 'dba':
        assert records[1]['backdoor_evaluation']['examples_seen'] == 4
    json.dumps(records, allow_nan=False)


def test_config_rejects_incompatible_rl_and_missing_root_before_loading():
    config = deepcopy(DEFAULTS)
    config['dataset']['name'] = 'cifar10'
    config['attack'].update(name='rl', malicious_clients=[0])
    with pytest.raises(ValueError, match='MNIST only'):
        run.build_experiment(config)
    config = deepcopy(DEFAULTS)
    config['defense']['name'] = 'fltrust'
    config['partition']['root_samples_per_class'] = 0
    with pytest.raises(ValueError, match='root set'):
        run.build_experiment(config)


@pytest.mark.parametrize('partition,defense,attack,parameters', [
    ('iid', 'fedavg', 'none', {}),
    ('dirichlet', 'clipmed', 'ipm', {'scaling_factor': 2.5}),
    ('iid', 'fltrust', 'dba', {'model_replacement_scale': 2.0}),
    ('dirichlet', 'median', 'bfl', {}),
])
def test_json_config_reaches_round_metrics(mnist_root, partition, defense, attack, parameters):
    config = small_config(mnist_root)
    config['partition']['method'] = partition
    config['defense']['name'] = defense
    if defense == 'clipmed':
        config['defense']['parameters'] = {'norm_bound': .04}
    config['attack'].update(name=attack, malicious_clients=[0], start_round=1, parameters=parameters)
    path = mnist_root / 'run.json'
    path.write_text(json.dumps(config))
    records = list(run.run_experiment(load_config(path)))
    assert [row['round'] for row in records] == [0, 1]
    assert records[1]['evaluation']['examples_seen'] == 10
    assert set(records[1]['selected_clients']) == {0, 1, 2}
    assert all(row['steps'] == 1 for row in records[1]['client_metrics'])
    if attack == 'ipm':
        assert records[1]['attack']['scaling_factor'] == 2.5
    if attack in {'dba', 'bfl'}:
        assert records[1]['attack']['malicious_clients'] == [0]
        assert records[1]['backdoor_evaluation']['examples_seen'] == (9 if attack == 'dba' else 1)
    if defense == 'fltrust':
        assert records[1]['root_metrics']['steps'] == 1
    json.dumps(records, allow_nan=False)


def test_partition_and_seeded_loaders_exclude_root_and_preserve_initialization(mnist_root):
    config = small_config(mnist_root)
    first, first_args = run.build_experiment(config)
    second, second_args = run.build_experiment(config)
    torch.testing.assert_close(model_state(first), model_state(second), rtol=0, atol=0)
    all_indices = []
    for a, b in zip(first_args['client_loaders'], second_args['client_loaders'], strict=True):
        all_indices.extend(a.dataset.indices)
        torch.testing.assert_close(next(iter(a)), next(iter(b)), rtol=0, atol=0)
    assert len(all_indices) == len(set(all_indices)) == 20
    assert first_args['learning_rate'] == .023


def test_tiny_assembly_uses_clean_root_transform_and_matching_trigger(tiny_root):
    config = small_config(tiny_root, dataset='tiny_imagenet')
    config['dataset']['options'] = {'train_per_class': 2, 'val_per_class': 1, 'normalization': 'tiny', 'augment': True}
    config['defense']['name'] = 'fltrust'
    config['attack'].update(name='dba', malicious_clients=[0], start_round=1)
    model, args = run.build_experiment(config)
    root = args['root_loader'].dataset
    clients = args['client_loaders']
    assert root.dataset.transform is args['test_loader'].dataset.transform
    assert root.dataset.transform is not clients[0].dataset.dataset.transform
    assert len(root.indices) == 200
    assert set(root.indices).isdisjoint(index for client in clients for index in client.dataset.indices)
    trigger = args['attack'].keywords['trigger']
    from meta_sg.datasets.tiny_imagenet.data import NORMALIZATION
    mean, std = NORMALIZATION['tiny']
    expected = (1 - torch.tensor(mean)) / torch.tensor(std)
    torch.testing.assert_close(trigger(torch.zeros(1, 3, 64, 64))[0, :, 0, 0], expected)
    metrics = args['backdoor_evaluation'](model)
    assert metrics['examples_seen'] == 199


def test_rl_config_keeps_attacker_learning_rate_and_same_model_seed(mnist_root, monkeypatch):
    class ActorFixture:
        def __init__(self, path):
            assert path == 'external-policy.zip'
            torch.rand(100)

    monkeypatch.setattr(run, 'FrozenLi2022Actor', ActorFixture)
    config = small_config(mnist_root)
    clean, _ = run.build_experiment(config)
    clean_rng = torch.get_rng_state()
    config['attack'].update(name='rl', malicious_clients=[0], parameters={
        'checkpoint': 'external-policy.zip', 'known_indices': [0, 1], 'attacker_learning_rate': .01,
    })
    attacked, args = run.build_experiment(config)
    torch.testing.assert_close(model_state(clean), model_state(attacked), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), clean_rng)
    assert args['learning_rate'] == .023
    assert args['attack'].keywords['attacker_learning_rate'] == .01
    assert args['attack'].keywords['known_images'].shape == (2, 1, 28, 28)


@pytest.mark.parametrize('attack', ['none', 'ipm', 'dba', 'bfl'])
def test_json_epoch_training_and_optimizer_parameters_reach_each_attack(mnist_root, attack):
    config = small_config(mnist_root)
    config['training'].update(rounds=2, local_steps=99, local_epochs=2, batch_size=4,
                              learning_rate=.004, learning_rate_decay=.5, momentum=.8, weight_decay=.001)
    config['attack'].update(name=attack, malicious_clients=[0], start_round=1)
    if attack == 'none':
        config['defense']['name'] = 'fltrust'
        config['training']['root_steps'] = 3
    if attack == 'bfl':
        config['attack']['parameters'] = {'source_label': None, 'target_label': 9, 'poison_fraction': .3125,
                                          'model_replacement_scale': 10, 'scale_by_malicious_count': True,
                                          'poison_selection': 'first'}
    path = mnist_root / 'epochs.json'
    path.write_text(json.dumps(config))
    model, arguments = run.build_experiment(load_config(path))
    for loader in arguments['client_loaders']:
        assert not loader.sampler.replacement
        assert sorted(iter(loader.sampler)) == list(range(len(loader.dataset)))
    records = list(run.run_federated(model, **arguments))
    assert [row['learning_rate'] for row in records] == [.004, .004, .002]
    for record in records[1:]:
        for client in record['client_metrics']:
            count = len(arguments['client_loaders'][client['client_id']].dataset)
            assert client['steps'] == 4
            assert client['examples_seen'] == 2 * count
        if attack == 'none':
            assert record['root_metrics']['steps'] == 3
        if attack == 'bfl':
            assert record['attack']['poison_fraction'] == .3125
            assert record['attack']['model_replacement_scale'] == 10
            assert record['backdoor_evaluation']['examples_seen'] == 9
    json.dumps(records, allow_nan=False)
