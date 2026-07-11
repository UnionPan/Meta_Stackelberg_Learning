"""Deterministic tiny PaperBSMG environment for reproducible scaled evidence."""

from __future__ import annotations

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


class DeterministicFourClientSampler:
    def sample(self, request, rng):
        del request, rng
        return (0, 1, 2, 3)


@dataclass
class DeterministicBenignTrainer:
    template: ModelState

    def train(self, client_id, state, rng):
        del state, rng
        delta = ModelState.from_tensors(
            np.full_like(tensor, 0.01 * (client_id + 1))
            for tensor in self.template.tensors
        )
        return ClientUpdate(client_id, delta, 4)


def deterministic_paper_dataset(seed: int = 4) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(8, 1, 8, 8, generator=generator)
    labels = (inputs.mean(dim=(1, 2, 3)) > 0).long()
    return TensorDataset(inputs, labels)


def make_deterministic_paper_env(
    *,
    seed: int,
    horizon: int = 2,
    task_id: str = 'paper-bsmg-scaled',
) -> PaperBSMGEnv:
    codec = TorchParameterCodec()
    with torch.random.fork_rng():
        torch.manual_seed(99)
        initial_model = TinyImageCNN()
    source = RandomSource(seed)
    state = RoundState(0, codec.capture(initial_model), source.capture())
    dataset = deterministic_paper_dataset()
    return PaperBSMGEnv(
        task_id=task_id,
        initial_state=state,
        rng=source,
        horizon=horizon,
        sample_size=4,
        initial_observed_max_norm=1.0,
        sampler=DeterministicFourClientSampler(),
        benign_trainer=DeterministicBenignTrainer(state.global_model),
        population=FixedMaliciousPopulation({0, 1}),
        model_factory=TinyImageCNN,
        codec=codec,
        attacker_dataset=dataset,
        attacker_num_examples={0: 4, 1: 4},
        root_dataset=dataset,
        observation_encoder=ModelTailObservationEncoder.from_model(initial_model),
        server_optimizer=ServerSGD(),
        local_search_learning_rate=0.01,
        local_search_batch_size=4,
        local_search_trajectories=1,
    )
