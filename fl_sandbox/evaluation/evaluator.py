"""Evaluation services for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def test_model(model, data_loader, device=None) -> tuple[float, float]:
    """Evaluate a classifier and return ``(loss, accuracy)``."""
    if torch is None:
        raise RuntimeError("torch is required to evaluate FL models")
    criterion = torch.nn.CrossEntropyLoss()
    runtime_device = device or next(model.parameters()).device
    correct, total, total_loss = 0, 0, 0.0
    model.to(runtime_device)
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(runtime_device)
            labels = labels.to(runtime_device)
            outputs = model(images)
            batch_size = labels.size(0)
            total_loss += criterion(outputs, labels).item() * batch_size
            predicted = outputs.argmax(dim=1)
            total += batch_size
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total else 0.0
    mean_loss = total_loss / total if total else 0.0
    return mean_loss, accuracy


@dataclass
class Evaluator:
    """Runs clean and optional poisoned evaluation for one global model."""

    clean_loader: object
    poisoned_loader: Optional[object] = None
    device: object = None

    def evaluate(self, model) -> tuple[float, float, float]:
        clean_loss, clean_acc = test_model(model, self.clean_loader, device=self.device)
        if self.poisoned_loader is None:
            return clean_loss, clean_acc, float("nan")
        _, backdoor_acc = test_model(model, self.poisoned_loader, device=self.device)
        return clean_loss, clean_acc, backdoor_acc
