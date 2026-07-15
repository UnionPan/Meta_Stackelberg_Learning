"""Stateless PyTorch local optimization for canonical client updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.core.model_state import model_difference
from meta_stackelberg.core.random_state import RandomSource
from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.federated.types import ClientUpdate, RoundState


class TorchLocalTrainer:
    """Create a fresh model/optimizer and return ``local - global`` for one client."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], torch.nn.Module],
        client_datasets: Mapping[int, Dataset],
        codec: TorchParameterCodec,
        learning_rate: float,
        local_epochs: int,
        batch_size: int,
        device: str | torch.device = 'cpu',
    ) -> None:
        if not math.isfinite(float(learning_rate)) or learning_rate < 0.0:
            raise ValueError('learning_rate must be finite and non-negative')
        if local_epochs <= 0:
            raise ValueError('local_epochs must be positive')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.model_factory = model_factory
        self.client_datasets = dict(client_datasets)
        self.codec = codec
        self.learning_rate = float(learning_rate)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {self.device} is not available')

    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate:
        if client_id not in self.client_datasets:
            raise KeyError(f'client {client_id} has no dataset')
        dataset = self.client_datasets[client_id]
        if len(dataset) <= 0:
            raise ValueError(f'client {client_id} dataset must not be empty')

        model = self.model_factory().to(self.device)
        self.codec.load(model, state.global_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=rng.torch,
        )

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        model.train()
        for _ in range(self.local_epochs):
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                examples = int(labels.shape[0])
                total_loss += float(loss.detach().item()) * examples
                total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
                total_seen += examples

        local_state = self.codec.capture(model)
        return ClientUpdate(
            client_id=client_id,
            delta=model_difference(local_state, state.global_model),
            num_examples=len(dataset),
            is_malicious=False,
            metadata={
                'train_loss': total_loss / total_seen,
                'train_accuracy': total_correct / total_seen,
            },
        )
