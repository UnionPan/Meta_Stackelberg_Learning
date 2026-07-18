from types import SimpleNamespace

import pytest

from meta_stackelberg.experiments.run_model_poisoning_evaluation import (
    _pretraining_domain_member,
    _record_matches,
    _reward_guard_decision,
    _validate_checkpoint,
    build_parser,
)


def _args(method):
    return SimpleNamespace(
        T=100, K=5, H=200, method=method,
        workers=100, untargeted_attackers=20, sample_size=10,
        post_defense_mode='neuroclip',
        materialize_mnist=False,
        parallel_clients=1,
    )


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
            'workers': 100,
            'untargeted_attackers': 20,
            'sample_size': 10,
            'protocol_signature': {
                'local_training_budget': (
                    'one-minibatch-step-per-local-iteration'
                ),
                'attack_client_sampling': (
                    'uniform-conditioned-on-at-least-one-benign-reference'
                ),
                'post_defense_mode': 'neuroclip',
                'mnist_input_pipeline': 'torchvision-transform-per-sample',
                'local_model_workspace': (
                    'reused-and-reset-per-client-update'
                ),
                'parallel_clients': 1,
            },
        },
    )


def test_evaluation_defaults_to_meta_sg() -> None:
    args = build_parser().parse_args([
        '--checkpoint', 'checkpoint.pt',
        '--data-root', 'data',
        '--output', 'output',
    ])
    assert args.method == 'meta-sg'
    assert args.post_defense_mode == 'neuroclip'
    assert args.materialize_mnist is False
    assert (args.parallel_clients, args.cpu_threads) == (1, 0)
    assert args.deterministic_torch is False
    assert (args.online_T, args.online_H, args.online_l) == (10, 100, 10)
    assert args.online_steps == 100
    assert args.online_batch_size == 256
    assert args.online_learning_starts == 100
    assert args.online_adaptation_step is None
    assert args.online_actor_logit_l2 == 0.0
    assert args.online_actor_logit_l2_mask is None
    assert args.online_selection == 'reward-guarded'
    assert args.online_selection_repeats == 2
    assert args.skip_online_adaptation is False
    assert (args.workers, args.untargeted_attackers, args.sample_size) == (
        100, 20, 10,
    )


def test_evaluation_can_select_only_scientific_gate_scenarios() -> None:
    args = build_parser().parse_args([
        '--checkpoint', 'checkpoint.pt',
        '--data-root', 'data',
        '--output', 'output',
        '--training-H', '2',
        '--skip-online-adaptation',
        '--scenarios', 'clean', 'ipm',
    ])
    assert args.scenarios == ['clean', 'ipm']
    assert args.training_H == 2
    assert args.skip_online_adaptation is True


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


def test_global_meta_rl_rl_scenarios_are_recorded_as_in_domain() -> None:
    scenario = SimpleNamespace(
        attack_family='rl', attacker_label='rl-krum',
    )
    assert _pretraining_domain_member(
        scenario,
        method='meta-rl',
        training_domain=('na', 'ipm', 'lmp', 'rl-krum', 'rl-clipmed'),
    )


def _cached_record(online_adaptation):
    return {
        'protocol': 'canonical-model-poisoning-final-evaluation-v1',
        'scenario': 'clean',
        'attack_family': 'clean',
        'fixed_attack_scale': None,
        'seed': 101,
        'horizon': 2,
        'defender_fingerprint': 'defender',
        'attacker_fingerprint': 'attacker',
        'method': 'meta-rl',
        'online_adaptation': online_adaptation,
        'round_metrics': [{}, {}],
    }


def test_frozen_evaluation_cache_with_null_online_record_is_reusable() -> None:
    scenario = SimpleNamespace(
        name='clean', attack_family='clean', scale=None,
    )
    assert _record_matches(
        _cached_record(None), scenario, 101, 2,
        'defender', 'attacker', 'meta-rl', None,
    )


def test_online_evaluation_cache_requires_adaptation_fingerprint() -> None:
    scenario = SimpleNamespace(
        name='clean', attack_family='clean', scale=None,
    )
    expected = {'adapted_defender_fingerprint': 'defender'}
    assert not _record_matches(
        _cached_record(None), scenario, 101, 2,
        'defender', 'attacker', 'meta-rl', expected,
    )
    assert _record_matches(
        _cached_record(expected), scenario, 101, 2,
        'defender', 'attacker', 'meta-rl', expected,
    )


def test_online_reward_guard_requires_strict_held_out_improvement() -> None:
    rejected = _reward_guard_decision(
        base_score=-1.0, adapted_score=-1.0, margin=0.0,
    )
    accepted = _reward_guard_decision(
        base_score=-1.0, adapted_score=-0.9, margin=0.01,
    )
    assert rejected['selected'] == 'frozen'
    assert accepted['selected'] == 'adapted'
