import dataclasses

import pytest

from meta_stackelberg.agents.td3 import PaperMetaSGConfig
from meta_stackelberg.experiments.paper_meta_sg import action_parameter_ledger


def test_paper_config_preserves_explicit_meta_sg_parameters() -> None:
    config = PaperMetaSGConfig()

    assert (config.T, config.K, config.l, config.N_A, config.N_D) == (100, 10, 10, 10, 10)
    assert (config.H_mnist, config.H_cifar) == (200, 500)
    assert (config.online_T, config.online_l) == (10, 10)
    assert (config.online_H_mnist, config.online_H_cifar) == (100, 200)
    assert config.online_steps == 100
    assert config.policy_learning_rate == 0.001
    assert config.td3_batch_size == 256
    assert config.gamma == 0.99
    assert (config.fl_batch_size, config.local_iterations, config.client_learning_rate) == (128, 1, 0.05)
    assert (config.workers, config.untargeted_attackers, config.subsampling_rate) == (100, 20, 0.1)
    assert (config.kappa, config.kappa_attacker, config.kappa_defender) == (0.001, 0.001, 0.001)
    assert config.meta_update_step == 1.0
    assert config.adaptation_step == 0.01
    assert config.rl_training_rounds == 300
    assert (config.full_fl_rounds_mnist, config.full_fl_rounds_cifar) == (500, 1000)
    assert config.backdoor_attackers == 5
    assert (config.root_samples_mnist, config.root_samples_cifar) == (100, 200)
    assert config.non_iid_q == 0.5
    assert (config.generated_seed_samples, config.generated_seed_q) == (200, 0.1)
    assert config.default_backdoor_reward_lambda == 0.5
    assert config.defender_action_dim == config.attacker_action_dim == 3
    assert config.state_encoder == 'last-two-learnable-blocks-v1'
    assert config.parameter_source('T') == 'paper-explicit'
    assert config.parameter_source('rl_training_rounds') == 'paper-explicit'
    assert config.parameter_source('tau') == 'sb3-compatible-declared'
    assert config.parameter_source('state_encoder') == 'paper-semantic-contract'
    assert {
        field.name: config.parameter_source(field.name)
        for field in dataclasses.fields(config)
    }


def test_scaled_config_changes_counts_but_not_semantics() -> None:
    paper = PaperMetaSGConfig()
    scaled = paper.scaled(
        T=2, K=2, H=8, l=2, N_A=2, N_D=2,
        workers=6, untargeted_attackers=3, sample_size=4,
        td3_batch_size=16, learning_starts=16, hidden_sizes=(64, 64),
        replay_capacity=4096,
    )

    assert (scaled.T, scaled.K, scaled.H, scaled.l, scaled.N_A, scaled.N_D) == (2, 2, 8, 2, 2, 2)
    assert scaled.paper_reference == paper
    assert scaled.defender_action_dim == scaled.attacker_action_dim == 3
    assert scaled.state_encoder == paper.state_encoder
    assert scaled.scale_provenance == 'scaled-conformance-only-v1'


def test_config_is_frozen_serializable_and_rejects_invalid_counts() -> None:
    config = PaperMetaSGConfig()
    payload = config.to_dict()
    assert PaperMetaSGConfig.from_dict(payload) == config
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.T = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match='T'):
        PaperMetaSGConfig(T=0)
    with pytest.raises(TypeError):
        config.scaled(T=1, defender_action_dim=2)  # type: ignore[call-arg]


def test_action_bound_ledger_exposes_every_non_paper_range_and_deviation() -> None:
    ledger = action_parameter_ledger()
    assert {(item.role, item.parameter) for item in ledger} == {
        ('defender', 'alpha'), ('defender', 'beta'), ('defender', 'epsilon'),
        ('attacker', 'gamma'), ('attacker', 'local_steps'),
        ('attacker', 'stealth_lambda'),
    }
    beta = next(item for item in ledger if item.parameter == 'beta')
    assert (beta.low, beta.high) == (0.0, 0.45)
    assert beta.source == 'implementation-declared'
    assert 'paper' in beta.deviation
    assert all(item.deviation for item in ledger)
