"""Check component replacement, real TD3 updates, frozen policies, and exact recovery."""

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from meta_sg.defenses.adaptive import adaptive_update
from meta_sg.experiments import run
from meta_sg.rl.observations import OBSERVATIONS, compact8, histogram
from meta_sg.rl.agent import TD3Agent
from meta_sg.rl.rewards import REWARDS, absolute_cost, root_loss_delta, utility_delta
from test_artifacts import augmented_mnist, configuration, read_metrics, small_model


def test_observation_summaries_and_reward_formulas_are_independent():
    context = {'features': np.arange(12).reshape(3, 4) / 10}
    np.testing.assert_array_equal(compact8(context), np.quantile(context['features'], [.5, .9], axis=0).ravel().astype(np.float32))
    summary = histogram(context)
    assert summary.shape == (40,)
    np.testing.assert_allclose(summary.reshape(4, 10).sum(1), 1)
    feedback = {'before': {'loss': 1., 'accuracy': .5}, 'after': {'loss': .8, 'accuracy': .6}, 'retained_risk': .2}
    assert root_loss_delta(feedback) == pytest.approx(.1)
    assert utility_delta(feedback, risk_weight=.5) == pytest.approx(.1)
    assert absolute_cost(feedback) == pytest.approx(-1.)


def test_three_action_mechanism_matches_weight_and_norm_formula():
    updates = [(torch.tensor([2., 0.]),), (torch.tensor([0., 4.]),)]
    context = {'norms': np.array([2., 4.]), 'root_norm': 2., 'risks': np.array([0., 8.]), 'score_limit': 8.}
    plain, _ = adaptive_update(updates, context, np.array([-1., -1., -1.]))
    torch.testing.assert_close(plain[0], torch.tensor([1., 2.]))
    defended, info = adaptive_update(updates, context, np.array([1., 1., 1.]))
    expected = torch.tensor([2., 2 * np.exp(-8)], dtype=torch.float32) / (1 + np.exp(-8))
    torch.testing.assert_close(defended[0], expected)
    assert info['scales'] == [1., .5]


def test_td3_updates_actor_and_critics_without_consuming_classifier_randomness():
    pytest.importorskip('stable_baselines3')
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    agent = TD3Agent(8, {'hidden_units': [8, 8], 'learning_starts': 0, 'batch_size': 2,
                         'buffer_size': 8, 'gradient_steps': 2}, 42)
    initial = deepcopy(agent.model.policy.state_dict())
    for index in range(4):
        observation = np.full(8, index / 10, dtype=np.float32)
        action = agent.action(observation)
        agent.observe(observation, action, float(index + 1), observation + .1, index == 3)
    assert agent.model._n_updates == 6
    trained = agent.model.policy.state_dict()
    for prefix in ['actor.mu', 'critic.qf0', 'critic.qf1']:
        assert any(not torch.equal(value, trained[key]) for key, value in initial.items() if key.startswith(prefix))
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def rl_configuration(root, output):
    config = configuration(root, output, 'ipm')
    config['training'].update(rounds=6, local_epochs=None, local_steps=1)
    config['defense']['name'] = 'rl'
    config['rl']['agent'] = {'learning_starts': 1, 'batch_size': 2, 'buffer_size': 3,
                             'gradient_steps': 2, 'hidden_units': [8, 8], 'policy_delay': 2}
    return config


def test_td3_training_resume_preserves_replay_targets_optimizers_and_pending_transition(mnist_root, monkeypatch):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.DATASETS, 'mnist', (augmented_mnist, run.DATASETS['mnist'][1]))
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, interrupted = mnist_root / 'rl-full', mnist_root / 'rl-interrupted'
    list(run.run_experiment(rl_configuration(mnist_root, full)))
    config = rl_configuration(mnist_root, interrupted)
    stream = run.run_experiment(config)
    for expected in range(6):
        assert next(stream)['round'] == expected
    stream.close()
    checkpoint = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert checkpoint['round'] == 4
    assert checkpoint['defender']['agent']['full']
    assert checkpoint['defender']['agent']['updates'] > 0
    assert checkpoint['defender']['pending'] is not None
    assert [row['round'] for row in run.run_experiment(config, resume=interrupted / 'latest.pt')] == [5, 6]
    assert read_metrics(full) == read_metrics(interrupted)
    a = torch.load(full / 'latest.pt', weights_only=True)
    b = torch.load(interrupted / 'latest.pt', weights_only=True)
    for name, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][name], rtol=0, atol=0)
    left, right = a['defender']['agent'], b['defender']['agent']
    for name, value in left['parameters']['policy'].items():
        torch.testing.assert_close(value, right['parameters']['policy'][name], rtol=0, atol=0)
    assert right['updates'] == left['updates'] > 0
    assert b['defender']['pending'] is None
    assert (interrupted / 'policy.zip').exists() and (interrupted / 'policy.json').exists()


def test_custom_components_receive_only_trusted_feedback(mnist_root, monkeypatch):
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    calls = []

    def observation(context, *, columns):
        assert set(context) == {'features', 'risks', 'norms', 'root_norm', 'before', 'score_limit'}
        calls.append('observation')
        return context['features'][:, columns].mean(0)

    def reward(feedback, *, value):
        assert set(feedback) == {'before', 'after', 'retained_risk'}
        calls.append('reward')
        return value

    monkeypatch.setitem(OBSERVATIONS, 'custom', observation)
    monkeypatch.setitem(REWARDS, 'custom', reward)
    config = rl_configuration(mnist_root, mnist_root / 'custom')
    config['training']['rounds'] = 2
    config['rl'].update(mode='fixed', observation={'name': 'custom', 'parameters': {'columns': [0, 2]}},
                         reward={'name': 'custom', 'parameters': {'value': .123}})
    records = list(run.run_experiment(config))
    assert calls == ['observation', 'reward'] * 2
    assert len(records[1]['aggregation']['observation']) == 2
    assert all(row['aggregation']['reward'] == .123 for row in records[1:])


def test_exported_policy_is_frozen_and_rejects_observation_mismatch(mnist_root, monkeypatch):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    output = mnist_root / 'policy-training'
    config = rl_configuration(mnist_root, output)
    config['training']['rounds'] = 3
    list(run.run_experiment(config))
    frozen = deepcopy(config)
    frozen['output']['directory'] = None
    frozen['rl'].update(mode='frozen', checkpoint=str(output / 'policy.zip'))
    model, arguments = run.build_experiment(frozen)
    defender = arguments['defender']
    before = deepcopy(defender.frozen.policy.state_dict())
    records = list(run.run_federated(model, **arguments))
    assert all(row['aggregation']['td3_updates'] == 0 for row in records[1:])
    for key, value in defender.frozen.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    frozen['rl']['observation'] = {'name': 'statistics', 'parameters': {}}
    with pytest.raises(ValueError, match='configuration must match'):
        run.build_experiment(frozen)
