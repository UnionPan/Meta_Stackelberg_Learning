"""Check archived observation/action semantics, signed SGD, and frozen policy IO."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from meta_sg.attacks.model_poisoning.rl import (
    FrozenLi2022Actor, archived_normalization, attacker_observation, decode_action,
    malicious_update,
)
from meta_sg.federated import model_state


def test_observation_uses_only_min_max_scaled_classifier_parameters():
    weight = torch.arange(1280).reshape(10, 128).float()
    bias = torch.arange(10).float() - 3
    observation = attacker_observation((torch.ones(20), weight, bias), 4)
    raw = np.concatenate([weight.numpy().ravel(), bias.numpy()])
    np.testing.assert_array_equal(observation['pram'], (2 * (raw - raw.min()) / np.ptp(raw) - 1).astype(np.float32))
    assert observation['num_attacker'] == 4
    assert not attacker_observation((weight * 0, bias * 0), 0)['pram'].any()


@pytest.mark.parametrize('action,expected', [([-1, -1], (0., 0)), ([0, 0], (5., 25)), ([1, 1], (10., 50))])
def test_action_decoding_matches_the_archive(action, expected):
    assert decode_action(action) == expected


def test_archived_normalization_matches_the_replacement_transform():
    raw = torch.arange(2400).remainder(256).to(torch.uint8).reshape(1200, 2)
    values = raw.float().div(255).sub(.1307).div(.3081)
    mean, std = archived_normalization(raw)
    assert mean == pytest.approx(float(values.mean()), abs=2e-7)
    assert std == pytest.approx(float(values.std()), abs=2e-7)


@pytest.mark.parametrize('rate,options', [(.01, {}), (.05, {'learning_rate': .05})],
                         ids=['archive-default', 'configured-005'])
def test_signed_update_matches_full_known_batch_sgd(rate, options):
    model = torch.nn.Linear(2, 2)
    state = model_state(model)
    images = torch.linspace(-1, 1, 400).reshape(200, 2)
    labels = torch.arange(200).remainder(2)
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=rate)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(reference(images), labels).backward()
        optimizer.step()
    actual, info = malicious_update(model, state, images, labels, gamma=5, steps=3,
                                   device='cpu', **options)
    for delta, trained, original in zip(actual, reference.parameters(), state, strict=True):
        torch.testing.assert_close(delta, -5 * (trained - original), rtol=1e-5, atol=2e-7)
    assert info['attacker_examples_seen'] == 600


def test_frozen_actor_preserves_checkpoint_predictions(tmp_path):
    sb3 = pytest.importorskip('stable_baselines3')
    gym = pytest.importorskip('gymnasium')

    class Env(gym.Env):
        observation_space = gym.spaces.Dict({
            'pram': gym.spaces.Box(-np.inf, np.inf, (1290,), dtype=np.float32),
            'num_attacker': gym.spaces.Discrete(11),
        })
        action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)

    original = sb3.TD3('MultiInputPolicy', Env(), policy_kwargs={'net_arch': [8, 4]}, device='cpu')
    checkpoint = tmp_path / 'policy.zip'
    original.save(checkpoint)
    frozen = FrozenLi2022Actor(checkpoint)
    state = (torch.arange(1280).reshape(10, 128).float(), torch.zeros(10))
    action, observation = frozen(state, 3)
    np.testing.assert_array_equal(action, original.predict(observation, deterministic=True)[0])
    assert not any(parameter.requires_grad for parameter in frozen.policy.policy.parameters())
