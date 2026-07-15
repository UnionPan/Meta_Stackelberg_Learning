"""Sample-mean clean classification evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from meta_stackelberg.core.model_state import ModelState
from meta_stackelberg.federated.models.parameters import TorchParameterCodec


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    num_examples: int


class ClassificationEvaluator:
    def __init__(
        self,
        *,
        model_factory: Callable[[], torch.nn.Module],
        dataset: Dataset,
        codec: TorchParameterCodec,
        batch_size: int,
        device: str | torch.device = 'cpu',
    ) -> None:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.model_factory = model_factory
        self.dataset = dataset
        self.codec = codec
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {self.device} is not available')

    def evaluate(self, state: ModelState) -> ClassificationMetrics:
        model = self.model_factory().to(self.device)
        self.codec.load(model, state)
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                examples = int(labels.shape[0])
                total_loss += float(loss.item()) * examples
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += examples
        if total_examples == 0:
            return ClassificationMetrics(loss=0.0, accuracy=0.0, num_examples=0)
        return ClassificationMetrics(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
            num_examples=total_examples,
        )
