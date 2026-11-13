"""Check interrupted-round recovery against uninterrupted synthetic runs."""

from copy import deepcopy
import json
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torchvision import transforms

from meta_sg.datasets.mnist import load_mnist
from meta_sg.experiments import run
from meta_sg.experiments.config import DEFAULTS


class RandomBrightness:
    def __call__(self, image):
        return image + .01 * (random.random() + np.random.random() + torch.rand(()))


def small_model():
    return nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(2, 10))


def augmented_mnist(root, **options):
    train, test = load_mnist(root, **options)
    train.transform = transforms.Compose([train.transform, RandomBrightness()])
    return train, test


def configuration(root, output, attack='none'):
    config = deepcopy(DEFAULTS)
    config['dataset']['data_root'] = str(root)
    config['partition'].update(method='dirichlet', root_samples_per_class=1)
    config['federation'].update(num_clients=3, clients_per_round=2)
    config['training'].update(rounds=4, local_epochs=2, batch_size=4, root_steps=2,
                              learning_rate=.01, learning_rate_decay=.8, momentum=.8, weight_decay=.01)
    config['output'].update(directory=str(output), checkpoint_interval=2)
    config['attack'].update(name=attack, malicious_clients=[0], start_round=2)
    if attack == 'none':
        config['defense']['name'] = 'fltrust'
    if attack == 'bfl':
        config['attack']['parameters'] = {'source_label': None, 'poison_fraction': .5,
                                          'model_replacement_scale': 2, 'scale_by_malicious_count': True}
    return config


def read_metrics(directory):
    return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('attack', ['none', 'ipm', 'dba', 'bfl'])
def test_recovery_matches_every_record_model_buffer_and_rng(mnist_root, monkeypatch, attack):
    monkeypatch.setitem(run.DATASETS, 'mnist', (augmented_mnist, run.DATASETS['mnist'][1]))
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, interrupted = mnist_root / 'full', mnist_root / 'interrupted'
    reference = json.loads(json.dumps(list(run.run_experiment(configuration(mnist_root, full, attack)))))
    reference_checkpoint = torch.load(full / 'latest.pt', weights_only=True)
    config = configuration(mnist_root, interrupted, attack)
    stream = run.run_experiment(config)
    for expected in range(4):
        assert next(stream)['round'] == expected
    stream.close()
    assert torch.load(interrupted / 'latest.pt', weights_only=True)['round'] == 2
    assert json.loads((interrupted / 'status.json').read_text())['phase'] == 'interrupted'
    with (interrupted / 'metrics.jsonl').open('a') as metrics:
        metrics.write('{"round":')

    def do_not_partition(*args, **kwargs):
        pytest.fail('resume must reuse saved partitions')

    monkeypatch.setattr(run, 'split_root_indices', do_not_partition)
    resumed = list(run.run_experiment(config, resume=interrupted / 'latest.pt'))
    assert [record['round'] for record in resumed] == [3, 4]
    assert read_metrics(interrupted) == reference
    actual = torch.load(interrupted / 'latest.pt', weights_only=True)
    for name, value in reference_checkpoint['model'].items():
        torch.testing.assert_close(actual['model'][name], value, rtol=0, atol=0)
    assert actual['participants'] == reference_checkpoint['participants']
    assert actual['random']['python'] == reference_checkpoint['random']['python']
    assert actual['random']['numpy'] == reference_checkpoint['random']['numpy']
    assert torch.equal(actual['random']['torch'], reference_checkpoint['random']['torch'])
    for a, b in zip(actual['loaders'], reference_checkpoint['loaders'], strict=True):
        for left, right in zip(a, b, strict=True):
            assert torch.equal(left, right)
    assert json.loads((interrupted / 'status.json').read_text())['phase'] == 'complete'
    assert not list(interrupted.glob('*.tmp'))


def test_resume_from_initial_checkpoint_and_extend_completed_budget(mnist_root, monkeypatch):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    output = mnist_root / 'initial'
    config = configuration(mnist_root, output)
    config['training']['rounds'] = 1
    stream = run.run_experiment(config)
    assert next(stream)['round'] == 0
    stream.close()
    assert [row['round'] for row in run.run_experiment(config, resume=output / 'latest.pt')] == [1]
    config['training']['rounds'] = 3
    assert [row['round'] for row in run.run_experiment(config, resume=output / 'latest.pt')] == [2, 3]
    assert [row['round'] for row in read_metrics(output)] == [0, 1, 2, 3]
    before = (output / 'metrics.jsonl').read_bytes()
    with pytest.raises(FileExistsError, match='empty output'):
        list(run.run_experiment(config))
    assert (output / 'metrics.jsonl').read_bytes() == before


def test_resume_checks_settings_and_input_contents_before_touching_metrics(mnist_root, monkeypatch):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    output = mnist_root / 'checked'
    config = configuration(mnist_root, output)
    stream = run.run_experiment(config)
    next(stream)
    stream.close()
    before = (output / 'metrics.jsonl').read_bytes()
    changed = deepcopy(config)
    changed['training']['learning_rate'] = .4
    with pytest.raises(ValueError, match='unchanged experiment settings'):
        list(run.run_experiment(changed, resume=output / 'latest.pt'))
    path = mnist_root / 'MNIST/raw/train-images-idx3-ubyte'
    data = bytearray(path.read_bytes())
    data[-1] ^= 255
    path.write_bytes(data)
    with pytest.raises(ValueError, match='dataset contents differ'):
        list(run.run_experiment(config, resume=output / 'latest.pt'))
    assert (output / 'metrics.jsonl').read_bytes() == before


def test_multiworker_augmentation_resumes_from_loader_rng(mnist_root, monkeypatch):
    monkeypatch.setitem(run.DATASETS, 'mnist', (augmented_mnist, run.DATASETS['mnist'][1]))
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, resumed = mnist_root / 'workers-full', mnist_root / 'workers-resumed'
    config = configuration(mnist_root, full)
    config['defense']['name'] = 'fedavg'
    config['federation'].update(num_clients=2, clients_per_round=1)
    config['training'].update(rounds=2, local_epochs=1, num_workers=1)
    config['output']['checkpoint_interval'] = 1
    list(run.run_experiment(config))
    config['output']['directory'] = str(resumed)
    stream = run.run_experiment(config)
    assert next(stream)['round'] == 0
    assert next(stream)['round'] == 1
    stream.close()
    list(run.run_experiment(config, resume=resumed / 'latest.pt'))
    assert read_metrics(full) == read_metrics(resumed)


def test_cli_saves_and_resumes_with_saved_configuration(mnist_root, monkeypatch, capsys):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    output = mnist_root / 'cli'
    config = configuration(mnist_root, output)
    config['training']['rounds'] = 1
    path = mnist_root / 'cli.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--config', str(path)])
    run.main()
    assert [json.loads(line)['round'] for line in capsys.readouterr().out.splitlines()] == [0, 1]
    config['training']['rounds'] = 2
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--config', str(path), '--resume', str(output / 'latest.pt')])
    run.main()
    assert [json.loads(line)['round'] for line in capsys.readouterr().out.splitlines()] == [2]
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--resume', str(output / 'latest.pt')])
    run.main()
    assert capsys.readouterr().out == ''
    assert json.loads((output / 'status.json').read_text())['phase'] == 'complete'


def test_training_failure_keeps_last_checkpoint_and_records_status(mnist_root, monkeypatch):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    output = mnist_root / 'failed'
    config = configuration(mnist_root, output)
    stream = run.run_experiment(config)
    next(stream)

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic interrupted round')

    monkeypatch.setattr(run, 'DEFENSES', {'fltrust': fail})
    stream.close()
    with pytest.raises(RuntimeError, match='synthetic interrupted round'):
        list(run.run_experiment(config, resume=output / 'latest.pt'))
    assert torch.load(output / 'latest.pt', weights_only=True)['round'] == 0
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'failed'
    assert status['checkpoint_round'] == 0
