from types import SimpleNamespace

import pytest

from meta_stackelberg.experiments.run_model_poisoning_evaluation import (
    _validate_checkpoint,
    build_parser,
)


def _args(method):
    return SimpleNamespace(T=100, K=5, H=200, method=method)


def _checkpoint(*, method, algorithm1, algorithm2):
    return SimpleNamespace(
        phase='complete',
        algorithm1_completed=algorithm1,
        algorithm2_completed=algorithm2,
        config_signature={
            'T': 100,
            'K': 5,
            'H': 200,
            'N_D': 10,
            'training_method': method,
        },
    )


def test_evaluation_defaults_to_meta_sg() -> None:
    args = build_parser().parse_args([
        '--checkpoint', 'checkpoint.pt',
        '--data-root', 'data',
        '--output', 'output',
    ])
    assert args.method == 'meta-sg'


def test_meta_sg_evaluation_requires_only_complete_algorithm1() -> None:
    _validate_checkpoint(
        _checkpoint(method='meta-sg', algorithm1=10, algorithm2=0),
        _args('meta-sg'),
    )


def test_meta_rl_checkpoint_cannot_be_evaluated_as_meta_sg() -> None:
    with pytest.raises(ValueError, match='method does not match'):
        _validate_checkpoint(
            _checkpoint(method='meta-rl', algorithm1=0, algorithm2=100),
            _args('meta-sg'),
        )


def test_meta_rl_evaluation_requires_complete_algorithm2() -> None:
    with pytest.raises(ValueError, match='Algorithm 2 is incomplete'):
        _validate_checkpoint(
            _checkpoint(method='meta-rl', algorithm1=0, algorithm2=99),
            _args('meta-rl'),
        )
