"""Check IPM values, Krum-targeted LMP selection, and Min-Max distance bounds."""

import pytest
import torch

from meta_sg.attacks.model_poisoning import ipm_updates, lmp_updates, min_max_updates
from meta_sg.defenses.krum import krum_scores
from meta_sg.federated.state import stack_updates


def test_ipm_is_the_scaled_negative_honest_mean_with_independent_submissions():
    honest = [(torch.tensor([1., 3.]),), (torch.tensor([3., 5.]),)]
    malicious, _ = ipm_updates(honest, num_malicious=2, scaling_factor=5)
    torch.testing.assert_close(malicious[0][0], torch.tensor([-10., -20.]))
    malicious[0][0].zero_()
    torch.testing.assert_close(malicious[1][0], torch.tensor([-10., -20.]))
    torch.testing.assert_close(honest[0][0], torch.tensor([1., 3.]))


@pytest.mark.parametrize('maximum', [0.5, 100.0])
def test_lmp_is_selected_by_its_krum_oracle_without_exceeding_the_search_cap(maximum):
    honest = [(torch.tensor([1. + offset, 1. - offset]),)
              for offset in (-.04, -.02, 0, .02, .04)]
    malicious, info = lmp_updates(honest, num_malicious=2, maximum_multiplier=maximum)
    selected = int(krum_scores(stack_updates(honest + malicious), 2).argmin())
    assert selected >= len(honest)
    assert info['selected_lmp'] and info['multiplier'] <= maximum
    torch.testing.assert_close(malicious[0], malicious[1], rtol=0, atol=0)


def test_lmp_allows_an_explicit_oracle_bound_with_one_selected_attacker():
    values = torch.randn(6, 10, generator=torch.Generator().manual_seed(42)) * .1 + 1
    honest = [(row,) for row in values]
    malicious, info = lmp_updates(honest, num_malicious=1, oracle_bound=2)
    assert len(malicious) == 1 and info['oracle_bound'] == 2
    assert int(krum_scores(stack_updates(honest + malicious), 2).argmin()) == len(honest)


def test_lmp_reports_a_search_failure_without_substituting_another_attack():
    with pytest.raises(RuntimeError, match='could not find an LMP candidate'):
        lmp_updates([(torch.zeros(5),)] * 8, num_malicious=2)


def test_min_max_std_preserves_the_gradient_to_delta_sign():
    honest = [(torch.tensor([-1., 0.]),), (torch.tensor([1., 0.]),)]
    malicious, info = min_max_updates(honest, num_malicious=2)
    torch.testing.assert_close(malicious[0][0], torch.tensor([1., 0.]), atol=3e-5, rtol=0)
    assert info['max_malicious_distance_sq'] <= 4
    malicious[0][0].zero_()
    assert malicious[1][0][0] > .99


@pytest.mark.parametrize('direction', ['std', 'sign', 'unit_vec'])
def test_min_max_stays_within_the_independent_pairwise_distance_bound(direction):
    vectors = torch.randn(16, 37, generator=torch.Generator().manual_seed(42)) + .3
    malicious, info = min_max_updates([(row.clone(),) for row in vectors],
                                     num_malicious=2, direction=direction)
    candidate = malicious[0][0]
    limit = max(float((left - right).square().sum()) for left in vectors for right in vectors)
    assert max(float((row - candidate).square().sum()) for row in vectors) <= limit + 1e-4
    mean = vectors.mean(dim=0)
    deviation = vectors.std(dim=0) if direction == 'std' else (
        -mean.sign() if direction == 'sign' else -mean / mean.norm())
    beyond = mean + (info['lambda'] + 1e-3) * deviation
    assert max(float((row - beyond).square().sum()) for row in vectors) > limit


def test_min_max_identical_references_return_the_mean():
    malicious, info = min_max_updates([(torch.ones(3),)] * 4, num_malicious=1)
    torch.testing.assert_close(malicious[0][0], torch.ones(3))
    assert info['degenerate']


def test_min_max_requires_two_honest_references():
    with pytest.raises(ValueError, match='two honest'):
        min_max_updates([(torch.ones(3),)], num_malicious=1)
