from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.experiments.attack_domain import (
    AttackTypeDomainSource,
    UniformAttackTypeSampler,
    load_attack_type_domain,
    save_attack_type_domain,
)


def _agent(seed):
    return TD3Agent(
        obs_dim=5, action_dim=3, role='attacker', seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_attack_domain_roundtrips_type_snapshots_and_defense_origins(tmp_path) -> None:
    policies = {'krum': _agent(1), 'clipmed': _agent(2)}
    source = AttackTypeDomainSource.from_policies(
        policies,
        origins={'krum': 'pretrained-against-krum',
                 'clipmed': 'pretrained-against-clipmed'},
    )
    path = tmp_path / 'attack-domain.pt'
    save_attack_type_domain(path, source)
    loaded = load_attack_type_domain(path)

    assert tuple(loaded.snapshots) == ('krum', 'clipmed')
    assert loaded.origins['krum'] == 'pretrained-against-krum'
    restored = _agent(9)
    restored.restore(loaded.snapshots['krum'])
    assert restored.fingerprint() == policies['krum'].fingerprint()


def test_attack_domain_rejects_missing_origin() -> None:
    try:
        AttackTypeDomainSource.from_policies(
            {'krum': _agent(1)}, origins={},
        )
    except ValueError as error:
        assert 'origins' in str(error)
    else:
        raise AssertionError('accepted attack type without origin provenance')


def test_uniform_attack_sampler_keeps_domain_size_independent_from_k() -> None:
    sampler = UniformAttackTypeSampler(('krum', 'clipmed'), seed=7)

    first = sampler(iteration=0, count=10)
    second = sampler(iteration=1, count=10)

    assert len(first) == len(second) == 10
    assert set(first) <= {'krum', 'clipmed'}
    assert set(second) <= {'krum', 'clipmed'}
    assert first != second
