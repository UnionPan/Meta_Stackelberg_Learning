"""Check Histogram40 statistics, frozen state, CLI fitting, and recovery."""

from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch

from meta_sg.experiments import run
from meta_sg.rl.distributions import DistributionTransform, FrozenDistributionObservation, summarize
from test_artifacts import read_metrics, small_model
from test_rl_defender import rl_configuration


def training_features():
    return np.random.default_rng(42).normal(size=(7, 13, 4)).astype(np.float32)


def test_histogram_layout_edges_tails_and_duplicate_boundaries():
    values = np.array([[-100, 100, -100, 100], [100, -100, -100, 100]])
    edges = np.zeros((9, 4))
    actual = summarize(values, 'histogram40', edges).reshape(10, 4)
    expected = np.zeros((10, 4), dtype=np.float32)
    expected[0] = [.5, .5, 1., 0.]
    expected[-1] = [.5, .5, 0., 1.]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(0), np.ones(4))
    assert np.count_nonzero(actual[1:-1]) == 0
    on_edge = summarize(np.zeros((3, 4)), 'histogram40', edges).reshape(10, 4)
    np.testing.assert_array_equal(on_edge[-1], np.ones(4))


@pytest.mark.parametrize('standardized', [False, True])
def test_fitted_statistics_round_trip_freeze_and_ignore_client_order(tmp_path, standardized):
    variant = 'histogram40'
    clients = training_features()
    transform = DistributionTransform.fit(clients, variant, standardized=standardized)
    np.testing.assert_array_equal(transform.edges, np.quantile(np.concatenate(clients), np.arange(.1, 1, .1), axis=0))
    summaries = np.stack([summarize(values, variant, transform.edges) for values in clients]).astype(float)
    expected_mean = summaries.mean(0) if standardized else np.zeros(summaries.shape[1])
    expected_scale = np.maximum(summaries.std(0), 1e-4) if standardized else np.ones(summaries.shape[1])
    np.testing.assert_array_equal(transform.mean, expected_mean)
    np.testing.assert_array_equal(transform.scale, expected_scale)
    path = tmp_path / 'transform.json'
    transform.save(path)
    loaded = DistributionTransform.load(path)
    before = loaded.to_dict()
    deployment = clients[0] * 100
    np.testing.assert_array_equal(loaded.encode(deployment), loaded.encode(deployment[::-1]))
    np.testing.assert_array_equal(loaded.encode(deployment), transform.encode(deployment))
    assert loaded.to_dict() == before
    assert len(transform.mean) == 40


def test_constant_calibration_uses_scale_floor_and_normalizes_to_zero():
    clients = np.ones((3, 4, 4), dtype=np.float32)
    transform = DistributionTransform.fit(clients, 'histogram40')
    np.testing.assert_array_equal(transform.scale, np.full(40, 1e-4))
    np.testing.assert_array_equal(transform.encode(clients[0]), np.zeros(40, dtype=np.float32))


def test_fit_cli_uses_only_training_archive_and_never_loads_fl_data(tmp_path, monkeypatch, capsys):
    variant = 'histogram40'
    path, output = tmp_path / 'training.npz', tmp_path / 'frozen.json'
    clients = training_features()
    np.savez(path, training_features=clients, evaluation_features=np.full_like(clients, 10000))

    def fail(*args, **kwargs):
        pytest.fail('fitting observation statistics must not build an FL experiment')

    monkeypatch.setattr(run, 'build_experiment', fail)
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--fit-observation', variant, '--calibration', str(path), '--output', str(output)])
    run.main()
    assert json.loads(capsys.readouterr().out)['dimensions'] == 40
    actual = DistributionTransform.load(output)
    expected = DistributionTransform.fit(clients, variant)
    assert actual.to_dict() == expected.to_dict()
    saved = json.loads(output.read_text())
    assert saved['training_rounds'] == 7 and saved['training_clients'] == 91
    assert saved['source_key'] == 'training_features'
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        run.main()
    assert output.read_bytes() == before


def test_signed_log_norm_is_observation_only_and_uses_legacy_float_precision(tmp_path):
    path = tmp_path / 'transform.json'
    transform = DistributionTransform.fit(training_features(), 'histogram40', standardized=False)
    transform.save(path)
    component = FrozenDistributionObservation('histogram40', transform_path=path, signed_log_norm=True)
    context = {'features': np.ones((3, 4), dtype=np.float64), 'norms': np.array([.5, 1., 2.])}
    before = context['features'].copy()
    expected = before.astype(np.float32)
    expected[:, 1] = np.log(context['norms'])
    np.testing.assert_array_equal(component(context), transform.encode(expected))
    np.testing.assert_array_equal(context['features'], before)


def test_frozen_distribution_survives_resume_and_export_without_source_file(mnist_root, monkeypatch):
    variant = 'histogram40'
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    path = mnist_root / 'transform.json'
    DistributionTransform.fit(training_features(), variant).save(path)
    full, interrupted = mnist_root / 'full', mnist_root / 'interrupted'
    config = rl_configuration(mnist_root, full)
    config['rl']['observation'] = {'name': variant, 'parameters': {'transform_path': str(path)}}
    list(run.run_experiment(config))
    config['output']['directory'] = str(interrupted)
    stream = run.run_experiment(config)
    for _ in range(6):
        next(stream)
    stream.close()
    state = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert state['defender']['observation']['variant'] == variant
    assert state['defender']['agent']['width'] == 40
    assert state['defender']['agent']['updates'] > 0
    path.unlink()

    def no_refit(*args, **kwargs):
        pytest.fail('resume and frozen inference must never refit observations')

    monkeypatch.setattr(DistributionTransform, 'fit', no_refit)
    list(run.run_experiment(config, resume=interrupted / 'latest.pt'))
    assert read_metrics(full) == read_metrics(interrupted)
    saved = json.loads((interrupted / 'policy.json').read_text())
    assert saved['observation_dim'] == 40
    assert saved['observation_state'] == json.loads((interrupted / 'observation.json').read_text())
    frozen = deepcopy(config)
    frozen['rl'].update(mode='frozen', checkpoint=str(interrupted / 'policy.zip'))
    frozen['output']['directory'] = None
    reference = deepcopy(saved['observation_state'])
    records = list(run.run_experiment(frozen))
    assert all(len(row['aggregation']['observation']) == 40 for row in records[1:])
    assert all(row['aggregation']['td3_updates'] == 0 for row in records[1:])
    assert json.loads((interrupted / 'policy.json').read_text())['observation_state'] == reference


def test_wrong_transform_variant_is_rejected(tmp_path):
    path = tmp_path / 'transform.json'
    state = DistributionTransform.fit(training_features(), 'histogram40').to_dict()
    state['variant'] = 'unsupported'
    path.write_text(json.dumps(state))
    component = FrozenDistributionObservation('histogram40', transform_path=path)
    with pytest.raises(ValueError, match='only Histogram40'):
        component({'features': training_features()[0]})


def test_metrics_transform_rejects_a_different_norm_channel(tmp_path):
    state = DistributionTransform.fit(training_features(), 'histogram40').to_dict()
    state['feature_norm'] = 'signed_log_norm'
    component = FrozenDistributionObservation('histogram40', transform_path=tmp_path / 'unused.json')
    with pytest.raises(ValueError, match='N feature definition differs'):
        component.load_state_dict(state)


def test_fit_cli_reads_archived_raw_key_without_standardization(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / 'raw.npz', tmp_path / 'histogram.json'
    np.savez(source, raw_client_features=training_features())
    monkeypatch.setattr(sys, 'argv', ['meta-sg', '--fit-observation', 'histogram40', '--calibration', str(source),
                                     '--calibration-key', 'raw_client_features', '--no-standardize', '--output', str(output)])
    run.main()
    assert json.loads(capsys.readouterr().out)['dimensions'] == 40
    state = json.loads(output.read_text())
    assert state['source_key'] == 'raw_client_features' and not state['standardized']
    np.testing.assert_array_equal(state['mean'], np.zeros(40))
    np.testing.assert_array_equal(state['scale'], np.ones(40))


@pytest.mark.parametrize('signed', [False, True])
def test_fit_from_training_metrics_connects_collection_to_frozen_training(mnist_root, monkeypatch, capsys, signed):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    source = mnist_root / 'calibration'
    config = rl_configuration(mnist_root, source)
    config['training']['rounds'] = 3
    config['rl']['mode'] = 'fixed'
    list(run.run_experiment(config))
    rows = read_metrics(source)
    clients = np.asarray([row['aggregation']['client_features'] for row in rows[1:]], dtype=np.float32)
    if signed:
        clients[:, :, 1] = np.log(np.asarray([row['aggregation']['client_norms'] for row in rows[1:]]))
    path = mnist_root / 'frozen.json'
    arguments = ['meta-sg', '--fit-observation', 'histogram40', '--calibration', str(source / 'metrics.jsonl'), '--output', str(path)]
    monkeypatch.setattr(sys, 'argv', arguments + (['--signed-log-norm'] if signed else []))
    run.main()
    assert json.loads(capsys.readouterr().out)['variant'] == 'histogram40'
    actual = DistributionTransform.load(path)
    assert actual.to_dict() == DistributionTransform.fit(clients, 'histogram40').to_dict()
    config['rl'].update(mode='train', observation={'name': 'histogram40', 'parameters': {'transform_path': str(path), 'signed_log_norm': signed}})
    config['output']['directory'] = str(mnist_root / 'trained')
    stream = run.run_experiment(config)
    assert next(stream)['round'] == 0
    stream.close()
    path.unlink()
    assert [row['round'] for row in run.run_experiment(config, resume=mnist_root / 'trained/latest.pt')] == [1, 2, 3]
    assert (mnist_root / 'trained/observation.json').exists()
