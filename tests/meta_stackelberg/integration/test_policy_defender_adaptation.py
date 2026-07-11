import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_adaptation import PolicyDefenderAdapter


def _agent(role, seed):
    return TD3Agent(
        obs_dim=3, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_defender_adaptation_uses_eta_steps_and_freezes_attacker() -> None:
    defender = _agent('defender', 1)
    attacker = _agent('attacker', 2)
    replay = TD3ReplayBuffer(
        16, obs_dim=3, action_dim=3, role='defender', seed=3,
    )
    defender_before = defender.fingerprint()
    attacker_before = attacker.fingerprint()

    def collect(adapted, frozen, target):
        del frozen
        for index in range(4):
            obs = np.full(3, index / 10, dtype=np.float32)
            target.add(
                obs, adapted.act(obs, deterministic=True), 1.0,
                obs + 0.01, False, generation=0, role='defender',
            )

    result = PolicyDefenderAdapter(
        l=2, batch_size=4, eta=0.01,
    ).adapt(
        defender=defender, attacker=attacker, replay=replay,
        collect_fresh=collect,
    )

    assert defender.fingerprint() == defender_before
    assert attacker.fingerprint() == attacker_before
    assert result.adapted_defender.fingerprint() != defender_before
    assert len(result.update_stats) == 2
    assert result.eta == 0.01
