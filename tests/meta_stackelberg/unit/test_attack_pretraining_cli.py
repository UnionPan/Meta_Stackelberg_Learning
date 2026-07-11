import pytest

from meta_stackelberg.experiments.run_attack_pretraining import (
    build_parser,
    make_pretraining_config,
    make_task_specs,
    validate_checkpoint_args,
)


def _args(*extra: str):
    return build_parser().parse_args([
        '--dataset', 'mnist', '--data-root', '/tmp/data',
        '--output', '/tmp/attack-domain.pt', '--clip-radius', '0.1',
        *extra,
    ])


def test_paper_pretraining_cli_requires_scale_acknowledgement() -> None:
    args = _args('--profile', 'paper')
    with pytest.raises(ValueError, match='allow-paper-scale'):
        make_pretraining_config(args)

    args.allow_paper_scale = True
    config, hidden_sizes = make_pretraining_config(args)
    assert config.fl_rounds == 300
    assert config.learning_starts == 100
    assert config.batch_size == 256
    assert hidden_sizes == (256, 256)


def test_cli_specs_separate_fixed_defense_parameters_from_meta_k() -> None:
    args = _args('--profile', 'micro')
    specs = make_task_specs(args, sample_size=10, attacker_fraction=0.2)

    assert [(item.label, item.defense) for item in specs] == [
        ('rl-krum', 'krum'), ('rl-clipmed', 'clipmed'),
    ]
    assert specs[0].byzantine_count == 2
    assert specs[1].clip_radius == 0.1
    assert not hasattr(args, 'K')


def test_resume_requires_checkpoint_directory() -> None:
    args = _args('--resume')
    with pytest.raises(ValueError, match='checkpoint-dir'):
        validate_checkpoint_args(args)

    args.checkpoint_dir = '/tmp/checkpoints'
    validate_checkpoint_args(args)
    assert args.checkpoint_interval == 25
