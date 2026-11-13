"""Check action replacement, selected controls, dimensions, export, and recovery."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from meta_sg.defenses.adaptive import adaptive_update
from meta_sg.experiments import run
from meta_sg.experiments.config import resolve_config
from meta_sg.rl.actions import ACTIONS, build_action, resolve_action
from test_artifacts import read_metrics, small_model
from test_rl_defender import rl_configuration


def sample_context():
    return {'features': np.array([[.1, 1., .01, -.1], [1.8, 2., .5, .2]]),
            'norms': np.array([2., 4.]), 'root_norm': 2., 'risks': np.array([0., 8.]), 'score_limit': 8.}


def test_selected_controls_are_expanded_by_name_in_configured_order():
    updates = [(torch.tensor([2., 0.]),), (torch.tensor([0., 4.]),)]
    spec = resolve_action({'name': 'mixed_norm', 'controls': ['strength', 'risk'], 'fixed': {'mix': -1.}})
    names, apply = build_action(spec)
    assert names == ('strength', 'risk')
    actual, info = apply(updates, sample_context(), np.array([.5, 1.]))
    expected, reference_info = adaptive_update(updates, sample_context(), np.array([1., -1., .5]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert info == reference_info


@pytest.mark.parametrize('name,mix', [('median_clip', -1.), ('trusted_norm', 1.)])
def test_two_control_components_match_the_existing_norm_modes(name, mix):
    updates = [(torch.tensor([2., 0.]),), (torch.tensor([0., 4.]),)]
    names, apply = build_action(resolve_action({'name': name}))
    actual, _ = apply(updates, sample_context(), np.array([1., .5]))
    expected, _ = adaptive_update(updates, sample_context(), np.array([1., mix, .5]))
    assert names == ('risk', 'strength')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cnb_uses_threshold_ranges_and_fourth_norm_control():
    updates = [(torch.tensor([2., 0.]),), (torch.tensor([0., 4.]),)]
    names, apply = build_action(resolve_action({'name': 'cnb'}))
    low, low_info = apply(updates, sample_context(), np.array([-1., 1., 0., -1.]))
    high, high_info = apply(updates, sample_context(), np.array([-1., 1., 0., 1.]))
    assert len(names) == 4
    assert low_info['thresholds'] == [.5, 6., 3.25]
    assert low_info['norm_bound_multiplier'] == .5
    assert high_info['norm_bound_multiplier'] == 5.
    assert low_info['scales'] == [.75, .375]
    assert high_info['scales'] == [1., 1.]
    assert high[0].norm() > low[0].norm()


@pytest.mark.parametrize('spec', [
    {'name': 'mixed_norm', 'controls': ['risk', 'risk'], 'fixed': {'mix': 0, 'strength': 1}},
    {'name': 'mixed_norm', 'controls': ['risk'], 'fixed': {'risk': 1, 'mix': 0, 'strength': 1}},
    {'name': 'mixed_norm', 'controls': ['risk'], 'fixed': {'mix': 0}},
    {'name': 'mixed_norm', 'controls': ['unknown'], 'fixed': {'mix': 0, 'strength': 1}},
])
def test_ambiguous_or_missing_control_assignments_are_rejected(spec):
    with pytest.raises(ValueError, match='partition'):
        resolve_action(spec)


def test_legacy_action_parameters_move_to_the_component_without_mutating_input():
    supplied = {'defense': {'name': 'rl', 'parameters': {'action': {'norm_cap': 1.5}}}}
    actual = resolve_config(supplied)
    assert actual['rl']['action']['parameters'] == {'norm_cap': 1.5}
    assert actual['defense']['parameters'] == {}
    assert supplied['defense']['parameters']['action'] == {'norm_cap': 1.5}


@pytest.mark.parametrize('spec,dimension', [
    ({'name': 'mixed_norm', 'controls': ['risk'], 'fixed': {'mix': -1., 'strength': 1.}}, 1),
    ({'name': 'median_clip'}, 2),
    ({'name': 'cnb'}, 4),
])
def test_dynamic_action_dimensions_train_resume_and_freeze(mnist_root, monkeypatch, spec, dimension):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, interrupted = mnist_root / 'full', mnist_root / 'interrupted'
    config = rl_configuration(mnist_root, full)
    config['rl']['action'] = spec
    list(run.run_experiment(config))
    config['output']['directory'] = str(interrupted)
    stream = run.run_experiment(config)
    for expected in range(6):
        assert next(stream)['round'] == expected
    stream.close()
    state = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert state['defender']['agent']['action_dimensions'] == dimension
    assert state['defender']['agent']['replay']['actions'].shape[-1] == dimension
    assert state['defender']['agent']['updates'] > 0
    list(run.run_experiment(config, resume=interrupted / 'latest.pt'))
    assert read_metrics(full) == read_metrics(interrupted)
    assert all(len(record['aggregation']['action']) == dimension for record in read_metrics(full)[1:])
    metadata = json.loads((interrupted / 'policy.json').read_text())
    assert len(metadata['actions']) == dimension
    frozen = deepcopy(config)
    frozen['output']['directory'] = None
    frozen['rl'].update(mode='frozen', checkpoint=str(interrupted / 'policy.zip'))
    records = list(run.run_experiment(frozen))
    assert all(len(row['aggregation']['action']) == dimension for row in records[1:])
    assert all(row['aggregation']['td3_updates'] == 0 for row in records[1:])
    incompatible = deepcopy(frozen)
    incompatible['rl']['action'] = {'name': 'mixed_norm'} if dimension != 3 else {'name': 'cnb'}
    with pytest.raises(ValueError, match='configuration must match'):
        list(run.run_experiment(incompatible))
    with pytest.raises(ValueError, match='unchanged experiment settings'):
        incompatible_resume = deepcopy(config)
        incompatible_resume['rl']['action'] = incompatible['rl']['action']
        list(run.run_experiment(incompatible_resume, resume=interrupted / 'latest.pt'))


def test_custom_action_function_uses_registered_names_without_changing_the_loop(mnist_root, monkeypatch):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    received = []

    def custom(updates, context, action, *, multiplier):
        received.append(action.copy())
        assert set(context) == {'features', 'risks', 'norms', 'root_norm', 'before', 'score_limit'}
        update, diagnostics = adaptive_update(updates, context, np.array([-1., -1., -1.]))
        return tuple(part * multiplier * action[0] for part in update), diagnostics

    monkeypatch.setitem(ACTIONS, 'custom', (('rate',), custom))
    config = rl_configuration(mnist_root, mnist_root / 'custom')
    config['training']['rounds'] = 2
    config['rl'].update(mode='fixed', fixed_action=[.5], action={'name': 'custom', 'parameters': {'multiplier': 2}})
    records = list(run.run_experiment(config))
    assert len(received) == 2
    assert all(values.tolist() == [.5] for values in received)
    assert records[1]['aggregation']['action_names'] == ['rate']


def test_previous_three_action_exports_and_checkpoints_still_load(mnist_root, monkeypatch):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, interrupted = mnist_root / 'legacy-full', mnist_root / 'legacy-interrupted'
    config = rl_configuration(mnist_root, full)
    config['defense']['parameters'] = {'action': {'norm_cap': 1.2}}
    list(run.run_experiment(config))
    config['output']['directory'] = str(interrupted)
    stream = run.run_experiment(config)
    for _ in range(6):
        next(stream)
    stream.close()
    metadata_path = interrupted / 'policy.json'
    metadata = json.loads(metadata_path.read_text())
    spec = metadata.pop('action')
    metadata['action_parameters']['action'] = spec['parameters']
    metadata_path.write_text(json.dumps(metadata))
    frozen = deepcopy(config)
    frozen['rl'].update(mode='frozen', checkpoint=str(interrupted / 'policy.zip'))
    frozen['output']['directory'] = None
    assert len(list(run.run_experiment(frozen))) == 7
    state = torch.load(interrupted / 'latest.pt', weights_only=True)
    state['config']['defense']['parameters']['action'] = state['config']['rl'].pop('action')['parameters']
    state['defender']['agent'].pop('action_dimensions')
    torch.save(state, interrupted / 'latest.pt')
    list(run.run_experiment(config, resume=interrupted / 'latest.pt'))
    assert read_metrics(full) == read_metrics(interrupted)
