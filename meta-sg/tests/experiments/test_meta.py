"""Verify Meta-SG update ordering, information boundaries, and exact recovery."""

from copy import deepcopy
import json
import random
import sys

import numpy as np
import pytest
import torch

from meta_sg.attacks.follower import decode_follower_action
from meta_sg.experiments import meta, run
from meta_sg.experiments.meta_game import FederatedMetaGame
from meta_sg.rl.agent import TD3Agent
from meta_sg.rl.distributions import DistributionTransform
from meta_sg.rl.meta import MetaSGLearner, first_order_update
from test_artifacts import augmented_mnist, read_metrics, small_model
from test_distributions import training_features
from test_rl_defender import rl_configuration


def meta_configuration(root, output):
    config = rl_configuration(root, output)
    config['attack']['name'] = 'none'
    config['federation']['clients_per_round'] = 3
    config['meta'].update(iterations=2, horizon=3, inner_steps=2, response_steps=2,
                          post_critic_steps=2, feedback_samples=10, tasks=[
        {'name': 'rl', 'kind': 'model_poisoning', 'malicious_clients': [0], 'start_round': 2},
        {'name': 'brl', 'kind': 'backdoor', 'malicious_clients': [0], 'start_round': 2,
         'parameters': {'source_label': None, 'target_label': 2}},
    ])
    return config


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def test_outer_gradient_applies_to_initial_actor_and_averages_only_critics():
    initial = {key: torch.tensor([value]) for key, value in
               [('actor.w', 10.), ('actor_target.w', 20.), ('critic.w', 30.), ('critic_target.w', 40.)]}
    adapted = [{key: value + offset for key, value in initial.items()} for offset in [2, 6]]
    updated = first_order_update(initial, adapted, [{'w': torch.tensor([2.])}, {'w': torch.tensor([4.])}], .5)
    assert updated['actor.w'].item() == 8.5
    assert updated['critic.w'].item() == 34.
    assert_same(updated['actor.w'], updated['actor_target.w'])
    assert_same(updated['critic.w'], updated['critic_target.w'])
    assert initial['actor.w'].item() == 10.


def test_critic_only_fit_and_actor_gradient_preserve_actor_targets_and_global_rng():
    pytest.importorskip('stable_baselines3')
    agent = TD3Agent(8, {'hidden_units': [8], 'batch_size': 2, 'buffer_size': 8}, 42)
    for index in range(4):
        obs = np.full(8, index / 10, dtype=np.float32)
        agent.add_transition(obs, np.ones(3, dtype=np.float32), 1., obs + .1, index == 3)
    before = agent.snapshot()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    report = agent.update(3, critic_only=True)
    gradients = agent.actor_loss_gradient()
    after = agent.snapshot()
    for key in before:
        if key.startswith(('actor.', 'actor_target.', 'critic_target.')):
            assert_same(before[key], after[key])
    assert any(not torch.equal(before[key], after[key]) for key in before if key.startswith('critic.'))
    assert sum(value.abs().sum() for value in gradients.values()) > 0
    assert report['actor_updates'] == 0 and report['critic_updates'] == 3
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert_same(torch.get_rng_state(), torch_state)


def test_inner_response_post_order_and_persistent_followers(tmp_path):
    pytest.importorskip('stable_baselines3')
    learner = MetaSGLearner(meta_configuration(tmp_path, tmp_path / 'out'), 8, 2, 42)
    base, phases, current, follower_before = learner.defender.snapshot(), [], {}, {}

    def rollout(task, defender, follower, horizon, seed, phase, episode, d_explore, f_explore):
        name = task['name']
        phases.append((name, phase))
        assert_same(learner.defender.snapshot(), base)
        if phase == 'inner_1':
            assert defender.samples == 0 and defender.model._n_updates == 0
            assert_same(defender.snapshot(), base)
            follower_before[name] = follower.snapshot()
        if phase == 'response_1':
            current[name] = defender.snapshot()
            assert_same(follower.snapshot(), follower_before[name])
        if phase.startswith('response'):
            assert_same(defender.snapshot(), current[name])
            assert (d_explore, f_explore) == (False, True)
        if phase == 'post_response':
            assert_same(defender.snapshot(), current[name])
            assert defender.samples == 0 and defender.model._n_updates == 0
            assert follower.model._n_updates == 2
        d_rows, f_rows = [], []
        for index in range(horizon):
            obs = np.full(8, .1 * (index + 1), dtype=np.float32)
            d_rows.append((obs, np.zeros(2, dtype=np.float32), 1., obs + .1, index == horizon - 1))
            f_rows.append((obs, np.zeros(3, dtype=np.float32), -1., obs + .1, index == horizon - 1))
        return {'defender': d_rows, 'follower': f_rows, 'metrics': {}}

    records, summary = learner.step(rollout)
    for name in ['rl', 'brl']:
        assert [phase for task, phase in phases if task == name] == ['inner_1', 'inner_2', 'response_1', 'response_2', 'post_response']
    assert summary['episodes'] == 10 and summary['fl_rounds'] == 30 and summary['actor_delta_norm'] > 0
    assert all(row['actor_updates'] == 0 for row in records if row['phase'] == 'post_response')
    restored = MetaSGLearner(meta_configuration(tmp_path, tmp_path / 'out'), 8, 2, 42)
    restored.load_state_dict(learner.state_dict())
    assert_same(restored.state_dict(), learner.state_dict())


def test_follower_control_extremes():
    assert decode_follower_action(np.array([-1., -1., -1.])) == (0., 0., 1)
    assert decode_follower_action(np.array([1., 1., 1.])) == (10., .1, 11)
    assert decode_follower_action(np.zeros(3)) == (5., .05, 6)


@pytest.mark.parametrize('histogram', [False, True])
def test_meta_full_pipeline_exact_outer_resume_and_frozen_export(mnist_root, monkeypatch, histogram):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    monkeypatch.setitem(run.DATASETS, 'mnist', (augmented_mnist, run.DATASETS['mnist'][1]))
    import meta_sg.federated.server as server

    def no_test_evaluation(*args, **kwargs):
        pytest.fail('meta optimization must not evaluate the test loader')

    monkeypatch.setattr(server, 'evaluate', no_test_evaluation)
    full, interrupted = mnist_root / 'full', mnist_root / 'interrupted'
    config = meta_configuration(mnist_root, full)
    if histogram:
        path = mnist_root / 'transform.json'
        DistributionTransform.fit(training_features(), 'histogram40').save(path)
        config['rl']['observation'] = {'name': 'histogram40', 'parameters': {'transform_path': str(path)}}
        config['rl']['action'] = {'name': 'mixed_norm', 'controls': ['strength'], 'fixed': {'risk': 1., 'mix': 0.}}
    reference = list(meta.run_meta(config))
    config['output']['directory'] = str(interrupted)
    stream = meta.run_meta(config)
    assert next(stream) == reference[0]
    stream.close()
    assert json.loads((interrupted / 'status.json').read_text())['phase'] == 'interrupted'
    for filename in ['metrics.jsonl', 'rounds.jsonl']:
        with (interrupted / filename).open('a') as file:
            file.write('{"unfinished":')
    if histogram:
        path.unlink()
    assert list(meta.run_meta(config, resume=interrupted / 'latest.pt')) == reference[1:]
    assert read_metrics(full) == read_metrics(interrupted)
    assert (full / 'rounds.jsonl').read_bytes() == (interrupted / 'rounds.jsonl').read_bytes()
    a = torch.load(full / 'latest.pt', weights_only=True)
    b = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert_same(a['learner'], b['learner'])
    assert_same(a['random'], b['random'])
    assert a['learner']['defender']['width'] == (40 if histogram else 8)
    assert a['learner']['defender']['action_dimensions'] == (1 if histogram else 3)
    assert all(state['width'] == 8 for state in b['learner']['followers'].values())
    rounds = [json.loads(line) for line in (full / 'rounds.jsonl').read_text().splitlines()]
    assert len(rounds) == reference[-1]['fl_rounds'] == 60
    for row in rounds:
        assert row['evaluation'] is None and row['backdoor_evaluation'] is None
        attack = row['attack']
        assert attack['follower_active'] == (row['round'] >= 2)
        expected = (attack['owned_target_accuracy'] or 0.) - attack['owned_clean_accuracy']
        assert attack['follower_reward'] == expected
    frozen = deepcopy(config)
    frozen['rl'].update(mode='frozen', checkpoint=str(interrupted / 'policy.zip'))
    _, arguments = run.build_experiment(frozen)
    assert arguments['defender'].frozen.observation_space.shape == ((40,) if histogram else (8,))
    assert json.loads((interrupted / 'status.json').read_text())['phase'] == 'complete'


def test_meta_dry_run_does_not_load_data_or_policy(tmp_path, monkeypatch, capsys):
    config = meta_configuration(tmp_path, tmp_path / 'out')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))

    def fail(*args, **kwargs):
        pytest.fail('dry run must not build episodes')

    monkeypatch.setattr(meta, 'FederatedMetaGame', fail)
    monkeypatch.setattr(sys, 'argv', ['meta-sg-meta', '--config', str(path), '--dry-run'])
    meta.main()
    report = json.loads(capsys.readouterr().out)
    assert report['resolved']['maximum_fl_rounds'] == 60
    config['attack']['name'] = 'ipm'
    with pytest.raises(ValueError, match='meta.tasks'):
        meta.describe_meta(config)


def test_mid_outer_interruption_rolls_back_partial_episodes(mnist_root, monkeypatch):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    full, interrupted = mnist_root / 'full', mnist_root / 'partial'
    config = meta_configuration(mnist_root, full)
    reference = list(meta.run_meta(config))
    original = FederatedMetaGame.__call__

    def interrupt(self, task, defender, follower, horizon, seed, phase, episode, *explore):
        result = original(self, task, defender, follower, horizon, seed, phase, episode, *explore)
        if episode == 13:
            raise KeyboardInterrupt('synthetic interruption during second outer iteration')
        return result

    config['output']['directory'] = str(interrupted)
    with monkeypatch.context() as patch:
        patch.setattr(FederatedMetaGame, '__call__', interrupt)
        with pytest.raises(KeyboardInterrupt):
            list(meta.run_meta(config))
    saved = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert saved['learner']['iteration'] == 1
    assert len((interrupted / 'rounds.jsonl').read_text().splitlines()) == 39
    assert list(meta.run_meta(config, resume=interrupted / 'latest.pt')) == reference[1:]
    for filename in ['metrics.jsonl', 'rounds.jsonl']:
        assert (interrupted / filename).read_bytes() == (full / filename).read_bytes()
    actual = torch.load(interrupted / 'latest.pt', weights_only=True)
    assert_same(actual['learner'], torch.load(full / 'latest.pt', weights_only=True)['learner'])


def test_owned_feedback_excludes_root(mnist_root, monkeypatch):
    pytest.importorskip('stable_baselines3')
    monkeypatch.setitem(run.MODELS['mnist'], 'cnn', small_model)
    config = meta_configuration(mnist_root, mnist_root / 'out')
    game = FederatedMetaGame(config)
    from meta_sg.attacks.follower import FollowerRound
    learner = MetaSGLearner(config, 8, 3, 42)
    for task in config['meta']['tasks']:
        follower = FollowerRound(task, learner.followers[task['name']], game.train, game.arguments['client_loaders'],
                                 game.evaluation_transform, game.trigger, 4, 10, 'cpu', 42, explore=False)
        root_indices = set(game.template.root_loader.dataset.indices)
        assert not root_indices.intersection(follower.owned_indices.tolist())
        assert set(follower.clean_loader.dataset.indices) <= set(follower.owned_indices.tolist())
        if follower.target_loader is not None:
            ids = follower.target_loader.dataset.indices
            assert set(ids) <= set(follower.owned_indices.tolist())
            assert torch.as_tensor(game.train.targets)[ids].ne(2).all()
