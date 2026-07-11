import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from meta_stackelberg.core.random_state import RandomSource


def test_patch_trigger_writes_exact_region_without_mutating_input() -> None:
    from meta_stackelberg.security.data.trigger import PatchTrigger

    trigger = PatchTrigger(row=1, column=2, height=2, width=2, value=7.0)
    image_2d = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    image_3d = torch.stack((image_2d, image_2d + 100.0))
    original_2d = image_2d.clone()
    original_3d = image_3d.clone()

    triggered_2d = trigger.apply(image_2d)
    triggered_3d = trigger.apply(image_3d)

    torch.testing.assert_close(image_2d, original_2d, rtol=0.0, atol=0.0)
    torch.testing.assert_close(image_3d, original_3d, rtol=0.0, atol=0.0)
    assert torch.all(triggered_2d[1:3, 2:4] == 7.0)
    assert torch.all(triggered_3d[:, 1:3, 2:4] == 7.0)
    outside = torch.ones_like(image_2d, dtype=torch.bool)
    outside[1:3, 2:4] = False
    torch.testing.assert_close(triggered_2d[outside], original_2d[outside], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'row': -1},
        {'column': -1},
        {'height': 0},
        {'width': 0},
        {'value': float('nan')},
    ],
)
def test_patch_trigger_rejects_invalid_configuration(kwargs) -> None:
    from meta_stackelberg.security.data.trigger import PatchTrigger

    values = {'row': 0, 'column': 0, 'height': 1, 'width': 1, 'value': 1.0}
    values.update(kwargs)
    with pytest.raises(ValueError):
        PatchTrigger(**values)


def test_patch_trigger_rejects_invalid_image_shape_or_bounds() -> None:
    from meta_stackelberg.security.data.trigger import PatchTrigger

    trigger = PatchTrigger(row=3, column=3, height=2, width=2, value=1.0)
    with pytest.raises(ValueError, match='bounds'):
        trigger.apply(torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match='2-D or 3-D'):
        trigger.apply(torch.zeros(1, 1, 4, 4))


def test_patch_trigger_rejects_value_not_representable_by_image_dtype() -> None:
    from meta_stackelberg.security.data.trigger import PatchTrigger

    with pytest.raises(ValueError, match='dtype'):
        PatchTrigger(row=0, column=0, height=1, width=1, value=0.5).apply(
            torch.zeros(2, 2, dtype=torch.int64)
        )


def _dataset() -> TensorDataset:
    images = torch.zeros(5, 1, 4, 4)
    images[:, :, 1, 1] = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor([0, 0, 1, 0, 1], dtype=torch.long)
    return TensorDataset(images, labels)


def _poisoned(fraction: float, seed: int = 13, dataset=None):
    from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
    from meta_stackelberg.security.data.trigger import PatchTrigger

    return SourceTargetPoisonedDataset(
        dataset=_dataset() if dataset is None else dataset,
        trigger=PatchTrigger(row=2, column=2, height=2, width=2, value=5.0),
        source_class=0,
        target_class=1,
        poison_fraction=fraction,
        rng=RandomSource(seed),
    )


def test_poison_selection_is_replayable_source_only_and_non_mutating() -> None:
    clean = _dataset()
    original_images = clean.tensors[0].clone()
    first = _poisoned(2.0 / 3.0, seed=17, dataset=clean)
    second = _poisoned(2.0 / 3.0, seed=17)

    assert first.poisoned_indices == second.poisoned_indices
    assert first.eligible_count == 3
    assert first.poisoned_count == 2
    assert first.poisoned_indices <= frozenset({0, 1, 3})
    for index in range(len(first)):
        poisoned_image, poisoned_label = first[index]
        clean_image, clean_label = _dataset()[index]
        if index in first.poisoned_indices:
            assert int(poisoned_label) == 1
            assert torch.all(poisoned_image[:, 2:4, 2:4] == 5.0)
        else:
            assert int(poisoned_label) == int(clean_label)
            torch.testing.assert_close(poisoned_image, clean_image, rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean.tensors[0], original_images, rtol=0.0, atol=0.0)


def test_zero_fraction_is_exact_clean_and_one_poisons_every_source() -> None:
    clean = _dataset()
    zero = _poisoned(0.0)
    full = _poisoned(1.0)

    assert zero.poisoned_indices == frozenset()
    assert full.poisoned_indices == frozenset({0, 1, 3})
    for index in range(len(clean)):
        zero_image, zero_label = zero[index]
        clean_image, clean_label = clean[index]
        torch.testing.assert_close(zero_image, clean_image, rtol=0.0, atol=0.0)
        assert int(zero_label) == int(clean_label)


@pytest.mark.parametrize('fraction', [-0.1, 1.1, float('nan')])
def test_poisoned_dataset_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match='poison_fraction'):
        _poisoned(fraction)


def test_poisoned_dataset_rejects_equal_classes_or_missing_source() -> None:
    from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
    from meta_stackelberg.security.data.trigger import PatchTrigger

    trigger = PatchTrigger(row=0, column=0, height=1, width=1, value=1.0)
    with pytest.raises(ValueError, match='different'):
        SourceTargetPoisonedDataset(
            dataset=_dataset(),
            trigger=trigger,
            source_class=1,
            target_class=1,
            poison_fraction=0.5,
            rng=RandomSource(1),
        )
    with pytest.raises(ValueError, match='source-class'):
        SourceTargetPoisonedDataset(
            dataset=_dataset(),
            trigger=trigger,
            source_class=9,
            target_class=1,
            poison_fraction=0.5,
            rng=RandomSource(1),
        )


@pytest.mark.parametrize('source_class,target_class', [(True, 1), (0.2, 1), (0, -1)])
def test_poisoned_dataset_rejects_invalid_class_ids(source_class, target_class) -> None:
    from meta_stackelberg.security.data.poisoning import SourceTargetPoisonedDataset
    from meta_stackelberg.security.data.trigger import PatchTrigger

    with pytest.raises((TypeError, ValueError)):
        SourceTargetPoisonedDataset(
            dataset=_dataset(),
            trigger=PatchTrigger(row=0, column=0, height=1, width=1, value=1.0),
            source_class=source_class,
            target_class=target_class,
            poison_fraction=0.5,
            rng=RandomSource(1),
        )
