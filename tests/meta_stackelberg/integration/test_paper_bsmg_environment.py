from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.environments.model_tail import ModelTailObservationEncoder
from meta_stackelberg.environments.paper_bsmg import PaperBSMGEnv
from meta_stackelberg.federated.engine.server_optimizer import ServerSGD
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.models.tiny_cnn import TinyImageCNN
from meta_stackelberg.federated.types import ClientUpdate, RoundState
from meta_stackelberg.security.population import FixedMaliciousPopulation


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
            np.full_like(tensor, 0.01 * (client_id + 1))
            for tensor in self.template.tensors
        )
        return ClientUpdate(client_id, delta, 4)


def _dataset() -> TensorDataset:
    generator = torch.Generator().manual_seed(4)
    inputs = torch.randn(8, 1, 8, 8, generator=generator)
    labels = (inputs.mean(dim=(1, 2, 3)) > 0).long()
    return TensorDataset(inputs, labels)


def _make_env(seed: int = 9) -> PaperBSMGEnv:
    codec = TorchParameterCodec()
    with torch.random.fork_rng():
        torch.manual_seed(99)
        initial_model = TinyImageCNN()
    source = RandomSource(seed)
    state = RoundState(0, codec.capture(initial_model), source.capture())
    return PaperBSMGEnv(
        task_id='paper-bsmg-smoke',
        initial_state=state,
        rng=source,
        horizon=2,
        sample_size=4,
        initial_observed_max_norm=1.0,
        sampler=FixedSampler(),
        benign_trainer=FixedTrainer(state.global_model),
        population=FixedMaliciousPopulation({0, 1}),
        model_factory=TinyImageCNN,
        codec=codec,
        attacker_dataset=_dataset(),
        attacker_num_examples={0: 4, 1: 4},
        root_dataset=_dataset(),
        observation_encoder=ModelTailObservationEncoder.from_model(initial_model),
        server_optimizer=ServerSGD(),
        local_search_learning_rate=0.01,
        local_search_batch_size=4,
        local_search_trajectories=1,
    )


def test_one_markov_step_has_two_sequential_3d_actions_and_exact_replay() -> None:
    defender_raw = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    attacker_raw = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    first = _make_env()
    replay = _make_env()

    first_pending = first.begin_round(defender_raw)
    replay_pending = replay.begin_round(defender_raw)
    assert first_pending.round_index == 0
    assert first_pending.defender_action.alpha == 0.5000005
    assert first_pending.attacker_observation['malicious_count'].tolist() == [2.0]
    first_step = first.finish_round(attacker_raw)
    replay_step = replay.finish_round(attacker_raw)

    assert first_step.transition.state_after.round_index == 1
    assert first_step.defender_raw_action.shape == first_step.attacker_raw_action.shape == (3,)
    assert first_step == replay_step
    assert first.state.round_index == 1


def test_epsilon_changes_post_defense_reward_but_not_intermediate_fl_state() -> None:
    low = _make_env()
    high = _make_env()
    attacker_raw = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    low.begin_round(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    high.begin_round(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    low_step = low.finish_round(attacker_raw)
    high_step = high.finish_round(attacker_raw)

    np.testing.assert_array_equal(
        low_step.transition.state_after.global_model.vector(),
        high_step.transition.state_after.global_model.vector(),
    )
    assert low_step.post_loss_after != high_step.post_loss_after
    assert low.final_delivered_model() is not None


def test_environment_rejects_overlapping_round_phases() -> None:
    env = _make_env()
    try:
        env.finish_round(np.zeros(3, dtype=np.float32))
    except RuntimeError:
        pass
    else:
        raise AssertionError('finished a round that was not begun')
    env.begin_round(np.zeros(3, dtype=np.float32))
    try:
        env.begin_round(np.zeros(3, dtype=np.float32))
    except RuntimeError:
        pass
    else:
        raise AssertionError('began an overlapping round')


def test_round_with_no_sampled_malicious_client_is_a_valid_no_attack_transition() -> None:
    env = _make_env()
    env.population = FixedMaliciousPopulation(set())
    pending = env.begin_round(np.zeros(3, dtype=np.float32))
    assert pending.attacker_observation['malicious_count'].tolist() == [0.0]

    step = env.finish_round(np.zeros(3, dtype=np.float32))

    assert step.transition.private_diagnostics['malicious_client_count'] == 0
    assert step.transition.state_after.round_index == 1
