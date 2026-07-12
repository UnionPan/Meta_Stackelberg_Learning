from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.paper_backdoor_bsmg import PaperBackdoorBSMGEnv
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.security.attacks.backdoor_action import BackdoorAction
from meta_stackelberg.security.data.mnist_global_trigger import mnist_global_trigger
from meta_stackelberg.security.population import FixedMaliciousPopulation


class BackdoorSmokeCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=3)
        self.fc = torch.nn.Linear(2, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.conv(inputs)).mean(dim=(2, 3))
        return self.fc(values)


def _model_factory() -> BackdoorSmokeCNN:
    with torch.random.fork_rng():
        torch.manual_seed(99)
        return BackdoorSmokeCNN()


class FixedSampler:
    def sample(self, request, rng):
        del request, rng
        return (0, 1, 2, 3)


@dataclass
class FixedTrainer:
    template: ModelState

    def train(self, client_id, state, rng):
        del state, rng
        delta = ModelState.from_tensors(
            np.full_like(tensor, 0.002 * (client_id + 1))
            for tensor in self.template.tensors
        )
        return ClientUpdate(client_id, delta, 4)


def _dataset(seed: int, offset: float = 0.0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(8, 1, 28, 28, generator=generator) * 0.1 + offset
    labels = torch.tensor([1, 1, 1, 1, 2, 3, 4, 5], dtype=torch.long)
    return TensorDataset(images, labels)


def _make_env(seed: int = 9) -> PaperBackdoorBSMGEnv:
    codec = TorchParameterCodec()
    initial_model = _model_factory()
    source = RandomSource(seed)
    state = RoundState(0, codec.capture(initial_model), source.capture())
    fixture = mnist_global_trigger()
    return PaperBackdoorBSMGEnv(
        task_id='mnist-whitebox-real-data-v1',
        initial_state=state,
        rng=source,
        horizon=2,
        sample_size=4,
        initial_observed_max_norm=1.0,
        sampler=FixedSampler(),
        benign_trainer=FixedTrainer(state.global_model),
        population=FixedMaliciousPopulation({0, 1}),
        model_factory=_model_factory,
        codec=codec,
        malicious_client_datasets={0: _dataset(10), 1: _dataset(11, 0.02)},
        reward_dataset=_dataset(12),
        observation_encoder=ModelTailObservationEncoder.from_model(initial_model),
        server_optimizer=ServerSGD(),
        trigger=fixture.trigger,
        source_class=fixture.source_class,
        target_class=fixture.target_class,
        malicious_batch_size=4,
        defender_lambda=0.5,
        attacker_lambda=0.5,
    )


def test_backdoor_bsmg_executes_defender_then_shared_brl_action() -> None:
    env = _make_env()
    defender_raw = np.zeros(3, dtype=np.float32)
    attacker_raw = np.zeros(3, dtype=np.float32)

    pending = env.begin_round(defender_raw)
    step = env.finish_round(attacker_raw)

    assert pending.attacker_observation['defender_action'].shape == (3,)
    assert pending.attacker_observation['malicious_count'].tolist() == [2.0]
    assert step.attacker_action == BackdoorAction(0.5, 0.05, 6)
    assert step.transition.state_after.round_index == 1
    assert step.defender_reward.source == 'mnist-whitebox-real-data-v1'
    assert step.attacker_reward.source == 'mnist-whitebox-real-data-v1'
    assert step.transition.private_diagnostics['malicious_client_count'] == 2
    assert all(
        update.metadata['poison_fraction'] == 0.5
        for update in step.transition.malicious_updates
    )


def test_backdoor_bsmg_replays_exactly_from_same_seed() -> None:
    first = _make_env()
    replay = _make_env()
    defender_raw = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    attacker_raw = np.array([0.0, 0.0, -0.9], dtype=np.float32)

    first.begin_round(defender_raw)
    replay.begin_round(defender_raw)
    first_step = first.finish_round(attacker_raw)
    replay_step = replay.finish_round(attacker_raw)

    assert first_step == replay_step
    np.testing.assert_array_equal(
        first.state.global_model.vector(), replay.state.global_model.vector(),
    )


def test_post_defense_does_not_mutate_intermediate_fl_state() -> None:
    low = _make_env()
    high = _make_env()
    attacker_raw = np.array([0.0, 0.0, -0.9], dtype=np.float32)

    low.begin_round(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    high.begin_round(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    low_step = low.finish_round(attacker_raw)
    high_step = high.finish_round(attacker_raw)

    np.testing.assert_array_equal(
        low_step.transition.state_after.global_model.vector(),
        high_step.transition.state_after.global_model.vector(),
    )
    assert low.final_delivered_model() is not None
    assert high.final_delivered_model() is not None


def test_backdoor_bsmg_enforces_round_phase_order() -> None:
    env = _make_env()

    try:
        env.finish_round(np.zeros(3, dtype=np.float32))
    except RuntimeError as error:
        assert 'begin_round' in str(error)
    else:
        raise AssertionError('finished a round before Defender commitment')

    env.begin_round(np.zeros(3, dtype=np.float32))
    try:
        env.begin_round(np.zeros(3, dtype=np.float32))
    except RuntimeError as error:
        assert 'already pending' in str(error)
    else:
        raise AssertionError('began overlapping backdoor rounds')
