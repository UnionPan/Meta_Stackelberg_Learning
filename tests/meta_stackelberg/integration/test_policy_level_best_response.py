import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import TD3ReplayBuffer
from meta_stackelberg.stackelberg.policy_response import PolicyBestResponseTrainer


def _agent(role: str, seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=2, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=1e-3, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def test_policy_response_updates_attacker_na_times_and_freezes_full_defender() -> None:
    defender = _agent('defender', 1)
    attacker = _agent('attacker', 2)
    replay = TD3ReplayBuffer(64, obs_dim=2, action_dim=3, role='attacker', seed=3)
    defender_before = defender.fingerprint()

    def collect(step: int) -> None:
        rng = np.random.default_rng(step + 10)
        for _ in range(4):
            obs = rng.normal(size=2).astype(np.float32)
            action = attacker.act(obs, deterministic=False)
            reward = float(action[0] + 0.5 * action[1])
            replay.add(obs, action, reward, obs + 0.1, False,
                       generation=step, role='attacker')

    trainer = PolicyBestResponseTrainer(N_A=3, batch_size=4)
    result = trainer.train(
        defender=defender,
        attacker=attacker,
        replay=replay,
        collect_fresh=collect,
        independent_objective=lambda policy: float(policy.act(
            np.array([0.2, -0.1], dtype=np.float32), deterministic=True,
        )[0]),
    )

    assert defender.fingerprint() == defender_before
    assert attacker.update_count == 3
    assert len(result.update_stats) == 3
    assert result.initial_attacker_fingerprint != result.adapted_attacker_fingerprint
    assert result.approximate_best_response.schema_version == 1


def test_policy_response_detects_defender_mutation_inside_collection() -> None:
    defender = _agent('defender', 4)
    attacker = _agent('attacker', 5)
    replay = TD3ReplayBuffer(16, obs_dim=2, action_dim=3, role='attacker', seed=6)

    def mutate(step):
        batch_obs = np.zeros(2, np.float32)
        for _ in range(4):
            replay.add(batch_obs, np.zeros(3), 0.0, batch_obs, False,
                       generation=step, role='attacker')
        with __import__('torch').no_grad():
            next(defender.actor.parameters()).add_(1.0)

    try:
        PolicyBestResponseTrainer(N_A=1, batch_size=4).train(
            defender=defender, attacker=attacker, replay=replay,
            collect_fresh=mutate, independent_objective=lambda policy: 0.0,
        )
    except RuntimeError as error:
        assert 'defender' in str(error)
    else:
        raise AssertionError('missed Defender mutation during BR')
