from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import RoundState
from meta_stackelberg.security.attacks.backdoor_action import BackdoorAction
from meta_stackelberg.security.attacks.rl_backdoor import RLBackdoorAttack
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger
from meta_stackelberg.security.types import RoundAttackContext


def _model_factory() -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(41)
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 10))


def _client_dataset(offset: float) -> TensorDataset:
    images = torch.zeros(6, 1, 28, 28, dtype=torch.float32) + offset
    labels = torch.tensor([1, 1, 1, 2, 3, 4], dtype=torch.long)
    return TensorDataset(images, labels)


def _attack() -> tuple[RLBackdoorAttack, RoundAttackContext]:
    codec = TorchParameterCodec()
    global_model = codec.capture(_model_factory())
    attack = RLBackdoorAttack(
        action=BackdoorAction(0.5, 0.05, 2),
        model_factory=_model_factory,
        codec=codec,
        client_datasets={3: _client_dataset(0.0), 7: _client_dataset(0.1)},
        trigger=mnist_global_trigger().trigger,
        source_class=1,
        target_class=7,
        batch_size=3,
    )
    context = RoundAttackContext(
        round_index=0,
        global_model=global_model,
        malicious_client_ids=(3, 7),
        benign_updates=(),
    )
    return attack, context


def test_rl_backdoor_uses_shared_action_for_all_malicious_clients() -> None:
    attack, context = _attack()

    updates = attack.craft_round(context, (RandomSource(101), RandomSource(202)))

    assert tuple(update.client_id for update in updates) == (3, 7)
    assert all(update.is_malicious for update in updates)
    assert all(update.metadata['attack_type'] == 'rl-backdoor' for update in updates)
    assert all(update.metadata['poison_fraction'] == 0.5 for update in updates)
    assert all(update.metadata['malicious_learning_rate'] == 0.05 for update in updates)
    assert all(update.metadata['malicious_local_epochs'] == 2 for update in updates)
    assert all(update.metadata['source_class'] == 1 for update in updates)
    assert all(update.metadata['target_class'] == 7 for update in updates)


def test_rl_backdoor_is_deterministic_for_recreated_client_rngs() -> None:
    attack, context = _attack()

    first = attack.craft_round(context, (RandomSource(101), RandomSource(202)))
    second = attack.craft_round(context, (RandomSource(101), RandomSource(202)))

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.delta.vector(), right.delta.vector())
        assert left.metadata == right.metadata


def test_rl_backdoor_requires_one_rng_and_dataset_per_malicious_client() -> None:
    attack, context = _attack()

    try:
        attack.craft_round(context, (RandomSource(101),))
    except ValueError as error:
        assert 'RNG count' in str(error)
    else:
        raise AssertionError('missing malicious-client RNG was accepted')


def test_rl_backdoor_records_zero_poisoning_when_client_has_no_source_class() -> None:
    codec = TorchParameterCodec()
    global_model = codec.capture(_model_factory())
    dataset = TensorDataset(
        torch.zeros(4, 1, 28, 28),
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
    )
    attack = RLBackdoorAttack(
        action=BackdoorAction(0.5, 0.05, 1),
        model_factory=_model_factory,
        codec=codec,
        client_datasets={3: dataset},
        trigger=mnist_global_trigger().trigger,
        source_class=1,
        target_class=7,
        batch_size=2,
    )
    context = RoundAttackContext(0, global_model, (3,), ())

    update = attack.craft_round(context, (RandomSource(101),))[0]

    assert update.is_malicious
    assert update.metadata['eligible_source_count'] == 0
    assert update.metadata['poisoned_count'] == 0
