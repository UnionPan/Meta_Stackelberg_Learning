import pytest

from meta_stackelberg.experiments.run_paper_evidence import (
    build_parser,
    make_profile_config,
    validate_attack_initialization,
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


def test_paper_profile_accepts_explicit_task_batch_override() -> None:
    args = build_parser().parse_args([
        '--dataset', 'mnist', '--profile', 'paper', '--allow-paper-scale',
        '--task-batch-size', '5',
        '--data-root', '/tmp/data', '--output', '/tmp/output',
    ])
    config = make_profile_config(args)
    assert (config.T, config.K, config.H) == (100, 5, 200)


def test_task_batch_override_must_be_positive() -> None:
    args = build_parser().parse_args([
        '--dataset', 'mnist', '--profile', 'paper', '--allow-paper-scale',
        '--K', '0',
        '--data-root', '/tmp/data', '--output', '/tmp/output',
    ])
    with pytest.raises(ValueError, match='task-batch-size'):
        make_profile_config(args)


def test_paper_profile_requires_pretrained_attack_domain_or_explicit_deviation() -> None:
    args = build_parser().parse_args([
        '--dataset', 'mnist', '--profile', 'paper', '--allow-paper-scale',
        '--data-root', '/tmp/data', '--output', '/tmp/output',
    ])
    with pytest.raises(ValueError, match='attack-domain'):
        validate_attack_initialization(args)
    args.attack_domain = '/tmp/domain.pt'
    validate_attack_initialization(args)
