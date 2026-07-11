import pytest

from meta_stackelberg.experiments.run_paper_evidence import (
    build_parser,
    make_profile_config,
)


def test_actor_active_cli_profile_keeps_distinct_algorithm_counts() -> None:
    args = build_parser().parse_args([
        '--dataset', 'mnist', '--profile', 'actor-active',
        '--data-root', '/tmp/data', '--output', '/tmp/output',
    ])
    config = make_profile_config(args)
    assert (config.T, config.K, config.H) == (1, 1, 1)
    assert (config.l, config.N_A, config.N_D) == (2, 2, 2)
    assert config.td3_batch_size == config.learning_starts == 2


def test_paper_profile_requires_explicit_scale_acknowledgement() -> None:
    args = build_parser().parse_args([
        '--dataset', 'cifar', '--profile', 'paper',
        '--data-root', '/tmp/data', '--output', '/tmp/output',
    ])
    with pytest.raises(ValueError, match='allow-paper-scale'):
        make_profile_config(args)
    args.allow_paper_scale = True
    config = make_profile_config(args)
    assert (config.T, config.K, config.H) == (100, 10, 500)
    assert (config.l, config.N_A, config.N_D) == (10, 10, 10)
    assert config.hidden_sizes == (256, 256)
