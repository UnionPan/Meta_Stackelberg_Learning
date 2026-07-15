"""Reference-[15] model-based local-search RL attack operator."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate
from meta_stackelberg.security.attacks.rl_action import RLAttackAction
from meta_stackelberg.security.types import AttackCapabilities, RoundAttackContext


class RLLocalSearchAttack:
    capabilities = AttackCapabilities(
        needs_global_model=True,
        needs_local_data=True,
        observes_benign_updates=True,
    )

    def __init__(
        self,
        *,
        action: RLAttackAction,
        model_factory,
        codec: TorchParameterCodec,
        local_dataset: Dataset,
        num_examples_by_client: Mapping[int, int],
        learning_rate: float,
        batch_size: int,
        trajectories: int,
        gradient_norm_cap: float | None = None,
    ) -> None:
        if not isinstance(action, RLAttackAction):
            raise TypeError('action must be RLAttackAction')
        if not isinstance(codec, TorchParameterCodec):
            raise TypeError('codec must be TorchParameterCodec')
        if len(local_dataset) <= 0:
            raise ValueError('local_dataset must not be empty')
        if isinstance(learning_rate, bool) or not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        for value, name in ((batch_size, 'batch_size'), (trajectories, 'trajectories')):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if gradient_norm_cap is not None and (
            isinstance(gradient_norm_cap, bool)
            or not math.isfinite(float(gradient_norm_cap))
            or gradient_norm_cap <= 0
        ):
            raise ValueError('gradient_norm_cap must be finite and positive')
        counts = dict(num_examples_by_client)
        if not counts or any(
            isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0
            or isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for client_id, count in counts.items()
        ):
            raise ValueError('num_examples_by_client must contain valid ids and counts')
        self.action = action
        self.model_factory = model_factory
        self.codec = codec
        self.local_dataset = local_dataset
        self.num_examples_by_client = counts
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.trajectories = trajectories
        self.gradient_norm_cap = (
            None if gradient_norm_cap is None else float(gradient_norm_cap)
        )

    def craft_round(
        self,
        context: RoundAttackContext,
        rngs: tuple[RandomSource, ...],
    ) -> tuple[ClientUpdate, ...]:
        if context.global_model is None:
            raise ValueError('RL local search requires the global model')
        if not context.benign_updates:
            raise ValueError('RL local search requires benign reference updates')
        if len(rngs) != len(context.malicious_client_ids):
            raise ValueError('RL local-search RNG count must match malicious clients')
        missing = set(context.malicious_client_ids) - set(self.num_examples_by_client)
        if missing:
            raise ValueError(f'no sample counts for malicious clients {sorted(missing)}')
        reference = _mean_update(context.benign_updates)
        primary_rng = rngs[0]
        endpoints = tuple(
            self._search_endpoint(context.global_model, reference, primary_rng)
            for _ in range(self.trajectories)
        )
        endpoint_mean = np.mean(np.stack([state.vector() for state in endpoints]), axis=0)
        global_vector = context.global_model.vector()
        canonical_delta = self.action.gamma * (endpoint_mean - global_vector)
        delta = _from_vector_like(canonical_delta, context.global_model)
        metadata = {
            'attack_type': 'rl-local-search',
            'gamma': self.action.gamma,
            'local_steps': self.action.local_steps,
            'stealth_lambda': self.action.stealth_lambda,
            'trajectory_count': self.trajectories,
            'paper_gradient_sign_converted_to_model_delta': True,
        }
        if self.gradient_norm_cap is not None:
            metadata['local_search_gradient_norm_cap'] = self.gradient_norm_cap
        return tuple(
            ClientUpdate(
                client_id=client_id,
                delta=delta,
                num_examples=self.num_examples_by_client[client_id],
                is_malicious=True,
                metadata=metadata,
            )
            for client_id in context.malicious_client_ids
        )

    def _search_endpoint(
        self,
        global_state: ModelState,
        benign_reference: np.ndarray,
        rng: RandomSource,
    ) -> ModelState:
        model = self.model_factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError('model_factory must return torch.nn.Module')
        self.codec.load(model, global_state)
        parameters = tuple(model.parameters())
        if len(global_state.tensors) < len(parameters):
            raise ValueError('global state omits learnable parameter tensors')
        global_parameter_vector = np.concatenate([
            tensor.reshape(-1) for tensor in global_state.tensors[:len(parameters)]
        ])
        parameter_size = global_parameter_vector.size
        global_flat = torch.tensor(
            global_parameter_vector,
            dtype=parameters[0].dtype,
            device=parameters[0].device,
        )
        benign = torch.tensor(
            benign_reference[:parameter_size],
            dtype=parameters[0].dtype,
            device=parameters[0].device,
        )
        for _ in range(self.action.local_steps):
            inputs, labels = self._sample_batch(rng, parameters[0].device)
            logits = model(inputs)
            empirical_loss = torch.nn.functional.cross_entropy(logits, labels)
            current = torch.cat([parameter.reshape(-1) for parameter in parameters])
            deviation = global_flat - current
            if float(torch.linalg.vector_norm(deviation).detach()) <= 1e-12:
                # Cosine is undefined at the exact global initialization. A
                # zero-valued, zero-gradient term lets empirical loss create
                # the first nonzero search direction without a 1/eps spike.
                cosine = current.sum() * 0.0
            else:
                cosine = torch.nn.functional.cosine_similarity(
                    deviation.unsqueeze(0), benign.unsqueeze(0), dim=1,
                    eps=1e-12,
                )[0]
            objective = (
                (1.0 - self.action.stealth_lambda) * empirical_loss
                + self.action.stealth_lambda * cosine
            )
            gradients = torch.autograd.grad(objective, parameters)
            gradient_norm = torch.linalg.vector_norm(torch.cat([
                gradient.detach().reshape(-1).float() for gradient in gradients
            ]))
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError('local-search objective produced a non-finite gradient')
            scale = 1.0 if self.gradient_norm_cap is None else min(
                1.0,
                self.gradient_norm_cap / max(float(gradient_norm), 1e-12),
            )
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter.add_(self.learning_rate * scale * gradient)
        return self.codec.capture(model)

    def _sample_batch(self, rng: RandomSource, device: torch.device):
        indices = rng.numpy.integers(
            0, len(self.local_dataset), size=self.batch_size, endpoint=False,
        )
        samples = [self.local_dataset[int(index)] for index in indices]
        inputs = torch.stack([sample[0] for sample in samples]).to(device)
        labels = torch.stack([
            torch.as_tensor(sample[1]) for sample in samples
        ]).long().to(device)
        return inputs, labels


def _mean_update(updates: tuple[ClientUpdate, ...]) -> np.ndarray:
    vectors = [update.delta.vector().astype(np.float64, copy=False) for update in updates]
    if any(vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError('benign update structures do not match')
    result = np.mean(np.stack(vectors), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError('benign reference must be finite')
    return result


def _from_vector_like(vector: np.ndarray, template: ModelState) -> ModelState:
    values = []
    offset = 0
    for tensor in template.tensors:
        end = offset + tensor.size
        values.append(vector[offset:end].reshape(tensor.shape).astype(tensor.dtype))
        offset = end
    return ModelState.from_tensors(values)
