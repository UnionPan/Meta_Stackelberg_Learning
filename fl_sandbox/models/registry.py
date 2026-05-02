"""Model factory used by the benchmark coordinator."""

from __future__ import annotations

try:
    from src.models.cnn import MNISTClassifier, ResNet18
except ImportError:  # pragma: no cover - models require the research stack today
    MNISTClassifier = None
    ResNet18 = None


def build_model(dataset: str):
    """Build the default model for a supported dataset."""
    if dataset == "cifar10":
        if ResNet18 is None:
            raise RuntimeError("ResNet18 is unavailable; check src.models.cnn")
        return ResNet18()
    if MNISTClassifier is None:
        raise RuntimeError("MNISTClassifier is unavailable; check src.models.cnn")
    return MNISTClassifier()
