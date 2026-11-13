"""Check normalized trigger composition, poisoning eligibility, scaling, and ASR."""

from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_sg.attacks.backdoor import bfl_round, evaluate_asr, poison_batch, train_backdoor_update
from meta_sg.datasets.mnist import dba_trigger
from meta_sg.datasets.mnist.triggers import DBA_PATTERNS
from meta_sg.federated import model_state


@pytest.mark.parametrize('patch', [None, 0, 1, 2, 3])
def test_normalized_trigger_matches_poisoning_raw_white_pixels_before_preprocessing(patch):
    raw = torch.zeros(2, 1, 28, 28, dtype=torch.uint8)
    expected = raw.clone()
    patterns = DBA_PATTERNS if patch is None else (DBA_PATTERNS[patch],)
    for pattern in patterns:
        for row, column in pattern:
            expected[:, :, row, column] = 255
    images = raw.float().div(255).sub(.1307).div(.3081)
    triggered = dba_trigger(images, trigger_index=patch)
    torch.testing.assert_close(triggered, expected.float().div(255).sub(.1307).div(.3081), rtol=0, atol=0)
    torch.testing.assert_close(images, raw.float().div(255).sub(.1307).div(.3081), rtol=0, atol=0)


def test_poisoning_changes_only_selected_non_target_examples():
    images = torch.zeros(10, 1, 28, 28)
    labels = torch.tensor([2, 2, 0, 1, 3, 4, 5, 6, 7, 8])
    poisoned, targets, selected = poison_batch(
        images, labels, trigger=partial(dba_trigger, trigger_index=1), target_label=2,
        poison_fraction=.5, generator=torch.Generator().manual_seed(7),
    )
    assert len(selected) == 5
    assert not torch.isin(selected, torch.tensor([0, 1])).any()
    assert targets[selected].eq(2).all()
    assert torch.count_nonzero(poisoned) == 5 * 4
    assert not images.any() and torch.equal(labels, torch.tensor([2, 2, 0, 1, 3, 4, 5, 6, 7, 8]))


def test_source_specific_poisoning_preserves_other_classes():
    images = torch.zeros(6, 1, 28, 28)
    labels = torch.tensor([0, 0, 1, 2, 9, 0])
    _, targets, selected = poison_batch(
        images, labels, trigger=dba_trigger, target_label=9, source_label=0,
        poison_fraction=1, generator=torch.Generator().manual_seed(42),
    )
    assert set(selected.tolist()) == {0, 1, 5}
    torch.testing.assert_close(targets, torch.tensor([9, 9, 1, 2, 9, 9]))


def test_model_replacement_scales_the_delta_after_backdoor_training():
    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 28, 28), torch.tensor([0, 1, 2, 0])), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    base = model_state(model)
    arguments = dict(trigger=dba_trigger, target_label=2, source_label=None,
                     poison_fraction=.5, steps=1, learning_rate=.01, device='cpu', seed=42)
    plain, info = train_backdoor_update(model, base, loader, model_replacement_scale=1, **arguments)
    scaled, _ = train_backdoor_update(model, base, loader, model_replacement_scale=10, **arguments)
    torch.testing.assert_close(scaled, tuple(10 * part for part in plain), rtol=0, atol=0)
    assert info['poisoned_examples'] == 2


class PixelClassifier(nn.Module):
    def forward(self, images):
        logits = images.new_zeros(len(images), 10)
        logits[:, 2] = images[:, 0, 10, 10]
        return logits


@pytest.mark.parametrize('source,expected,count', [(None, .5, 4), (0, 1., 2)])
def test_asr_excludes_target_examples_and_applies_the_source_filter(source, expected, count):
    images = torch.zeros(6, 1, 28, 28)
    images[:, 0, 10, 10] = torch.tensor([1., 1., 1., -1., 1., -1.])
    labels = torch.tensor([2, 2, 0, 1, 0, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    result = evaluate_asr(PixelClassifier(), loader, trigger=dba_trigger,
                          target_label=2, source_label=source, device='cpu')
    assert result == {'asr': expected, 'examples_seen': count}


def test_asr_reports_an_undefined_denominator():
    loader = DataLoader(TensorDataset(torch.zeros(2, 1, 28, 28), torch.full((2,), 2)))
    with pytest.raises(ValueError, match='eligible'):
        evaluate_asr(PixelClassifier(), loader, trigger=dba_trigger, target_label=2, device='cpu')


def test_first_selection_preserves_batch_order_and_eligible_filter():
    images = torch.zeros(16, 1, 28, 28)
    labels = torch.tensor([9, 0, 1, 9, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4])
    _, targets, selected = poison_batch(images, labels, trigger=dba_trigger, target_label=9,
                                        poison_fraction=.3125, poison_selection='first')
    assert selected.tolist() == [1, 2, 4, 5, 6]
    assert targets[selected].eq(9).all()
    torch.testing.assert_close(labels, torch.tensor([9, 0, 1, 9, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4]))


@pytest.mark.parametrize('selected,expected_scale', [([0, 1, 2], 5.), ([0, 1], 10.), ([1], 10.)])
def test_bfl_configures_poisoning_and_scales_by_active_selected_attackers(selected, expected_scale):
    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    base = model_state(model)
    arguments = dict(round_index=1, steps=1, learning_rate=.02, device='cpu', seed=42,
                     malicious_clients=[0, 2], trigger=dba_trigger, target_label=9, source_label=None,
                     poison_fraction=.25, poison_selection='first', malicious_learning_rate=.03)
    plain, _, _ = bfl_round(model, base, [loader] * 3, selected,
                            model_replacement_scale=1, **arguments)
    scaled, metrics, info = bfl_round(model, base, [loader] * 3, selected,
                                      model_replacement_scale=10, scale_by_malicious_count=True, **arguments)
    assert info['model_replacement_scale'] == expected_scale
    assert info['malicious_learning_rate'] == .03
    for client in selected:
        if client in [0, 2]:
            torch.testing.assert_close(scaled[client], tuple(expected_scale * part for part in plain[client]), rtol=0, atol=0)
            row = next(row for row in metrics if row['client_id'] == client)
            assert row['poisoned_examples'] == 1
        else:
            torch.testing.assert_close(scaled[client], plain[client], rtol=0, atol=0)
