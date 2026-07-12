from __future__ import annotations

import pytest

from meta_stackelberg.experiments.run_mnist_whitebox import (
    build_parser,
    make_meta_config,
    make_pretraining_config,
)


def test_whitebox_cli_has_separate_pretrain_and_meta_sg_stages() -> None:
    parser = build_parser()
    pretrain = parser.parse_args([
        'pretrain', '--data-root', '/tmp/data', '--output', '/tmp/domain.pt',
    ])
    meta = parser.parse_args([
        'meta-sg', '--data-root', '/tmp/data',
        '--attack-domain', '/tmp/domain.pt', '--output', '/tmp/run',
    ])
    scientific = parser.parse_args([
        'scientific', '--data-root', '/tmp/data',
        '--attack-domain', '/tmp/domain.pt',
        '--policy-artifact', '/tmp/policies.pt', '--output', '/tmp/evidence',
    ])

    assert pretrain.command == 'pretrain'
    assert pretrain.norm_clip_radius == 1.0
    assert pretrain.neuroclip_epsilon == 7.0
    assert meta.command == 'meta-sg'
    assert scientific.command == 'scientific'
    assert scientific.clean_accuracy_floor == 0.8
    assert scientific.asr_ceiling == 0.2
    assert scientific.asr_reduction == 0.2


def test_whitebox_micro_profiles_are_execution_only() -> None:
    parser = build_parser()
    pretrain = parser.parse_args([
        'pretrain', '--profile', 'micro',
        '--data-root', '/tmp/data', '--output', '/tmp/domain.pt',
    ])
    meta = parser.parse_args([
        'meta-sg', '--profile', 'micro',
        '--data-root', '/tmp/data',
        '--attack-domain', '/tmp/domain.pt', '--output', '/tmp/run',
    ])

    pretraining_config, hidden_sizes = make_pretraining_config(pretrain)
    meta_config = make_meta_config(meta)

    assert pretraining_config.fl_rounds == 2
    assert hidden_sizes == (8,)
    assert (meta_config.N_D, meta_config.K, meta_config.N_A, meta_config.H) == (
        1, 1, 1, 1,
    )
    assert meta.execution_only is True


@pytest.mark.parametrize('command', ['pretrain', 'meta-sg'])
def test_whitebox_paper_profile_requires_explicit_scale_acknowledgement(
    command: str,
) -> None:
    parser = build_parser()
    arguments = [command, '--profile', 'paper', '--data-root', '/tmp/data']
    if command == 'pretrain':
        arguments += ['--output', '/tmp/domain.pt']
    else:
        arguments += [
            '--attack-domain', '/tmp/domain.pt', '--output', '/tmp/run',
        ]
    args = parser.parse_args(arguments)

    with pytest.raises(ValueError, match='allow-paper-scale'):
        if command == 'pretrain':
            make_pretraining_config(args)
        else:
            make_meta_config(args)
