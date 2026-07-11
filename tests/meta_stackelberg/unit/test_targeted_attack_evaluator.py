import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.federated.models.parameters import TorchParameterCodec
from meta_stackelberg.security.data.trigger import PatchTrigger


class PatchDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[..., -1, -1].reshape(inputs.shape[0]) * self.scale
        return torch.stack((1.0 - score, score), dim=1)


def _dataset() -> TensorDataset:
    images = torch.zeros((5, 1, 2, 2), dtype=torch.float32)
    images[1, ..., -1, -1] = 0.25
    labels = torch.tensor([0, 0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(images, labels)


@pytest.mark.parametrize('batch_size', [1, 2, 5])
def test_targeted_evaluator_uses_held_out_source_only_denominator(batch_size: int) -> None:
    from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator

    model = PatchDetector()
    codec = TorchParameterCodec()
    dataset = _dataset()
    original_images = dataset.tensors[0].clone()
    state = codec.capture(model)
    original_state = state.vector().copy()

    result = TargetedAttackEvaluator(
        model_factory=PatchDetector,
        dataset=dataset,
        codec=codec,
        trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
        source_class=0,
        target_class=1,
        batch_size=batch_size,
    ).evaluate(state)

    assert result.successes == 3
    assert result.source_examples == 3
    assert result.attack_success_rate == 1.0
    torch.testing.assert_close(dataset.tensors[0], original_images)
    np.testing.assert_array_equal(state.vector(), original_state)


def test_targeted_evaluator_rejects_empty_source_denominator() -> None:
    from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator

    dataset = TensorDataset(
        torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        torch.ones(2, dtype=torch.long),
    )
    codec = TorchParameterCodec()
    evaluator = TargetedAttackEvaluator(
        model_factory=PatchDetector,
        dataset=dataset,
        codec=codec,
        trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
        source_class=0,
        target_class=1,
        batch_size=2,
    )

    with pytest.raises(ValueError, match='source-class'):
        evaluator.evaluate(codec.capture(PatchDetector()))


@pytest.mark.parametrize('batch_size', [0, -1])
def test_targeted_evaluator_rejects_nonpositive_batch_size(batch_size: int) -> None:
    from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator

    with pytest.raises(ValueError, match='batch_size'):
        TargetedAttackEvaluator(
            model_factory=PatchDetector,
            dataset=_dataset(),
            codec=TorchParameterCodec(),
            trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
            source_class=0,
            target_class=1,
            batch_size=batch_size,
        )


def test_targeted_evaluator_requires_distinct_source_and_target() -> None:
    from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator

    with pytest.raises(ValueError, match='differ'):
        TargetedAttackEvaluator(
            model_factory=PatchDetector,
            dataset=_dataset(),
            codec=TorchParameterCodec(),
            trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
            source_class=0,
            target_class=0,
            batch_size=2,
        )


def test_targeted_evaluator_accepts_python_integer_labels() -> None:
    from meta_stackelberg.evaluation.targeted import TargetedAttackEvaluator

    class PythonLabelDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros((1, 2, 2)), (0 if index == 0 else 1)

    codec = TorchParameterCodec()
    metrics = TargetedAttackEvaluator(
        model_factory=PatchDetector,
        dataset=PythonLabelDataset(),
        codec=codec,
        trigger=PatchTrigger(row=1, column=1, height=1, width=1, value=1.0),
        source_class=0,
        target_class=1,
        batch_size=2,
    ).evaluate(codec.capture(PatchDetector()))

    assert metrics.source_examples == 1
    assert metrics.attack_success_rate == 1.0
