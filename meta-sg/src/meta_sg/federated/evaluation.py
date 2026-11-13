"""Design: evaluate any classifier using preprocessed DataLoader batches.
Data flow: batches -> logits -> summed cross-entropy and correct predictions ->
mean loss and accuracy. Count actual examples, including an incomplete last batch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    total_correct = 0
    examples_seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels)
        total_correct += logits.argmax(dim=1).eq(labels).sum()
        examples_seen += len(labels)
    return {
        'loss': float(total_loss) / examples_seen,
        'accuracy': float(total_correct) / examples_seen,
        'examples_seen': examples_seen,
    }
