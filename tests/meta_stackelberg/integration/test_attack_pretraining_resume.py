import numpy as np

from meta_stackelberg.agents.td3.agent import TD3Agent
from meta_stackelberg.agents.td3.replay import flatten_observation
from meta_stackelberg.experiments.attack_pretraining import (
    AttackPolicyPretrainingConfig,
    AttackPolicyPretrainer,
    fixed_pretraining_aggregator,
)
from meta_stackelberg.experiments.attack_pretraining_checkpoint import (
    load_attack_pretraining_checkpoint,
    save_attack_pretraining_checkpoint,
)
from meta_stackelberg.experiments.deterministic_paper_env import (
    make_deterministic_paper_env,
)
from meta_stackelberg.experiments.paper_meta_sg import (
    ATTACKER_OBSERVATION_KEYS,
    DEFENDER_OBSERVATION_KEYS,
)


def _agent(obs_dim: int, role: str, seed: int) -> TD3Agent:
    return TD3Agent(
        obs_dim=obs_dim, action_dim=3, role=role, seed=seed,
        hidden_sizes=(8,), learning_rate=0.001, gamma=0.99, tau=0.005,
        policy_delay=2, target_policy_noise=0.2, noise_clip=0.5,
    )


def _dimensions() -> tuple[int, int]:
    env = make_deterministic_paper_env(seed=30, horizon=6)
    defender_dim = len(flatten_observation(
        env.defender_observation(), DEFENDER_OBSERVATION_KEYS,
    ))
    attacker_dim = len(flatten_observation(
        env.begin_round(np.zeros(3, dtype=np.float32)).attacker_observation,
        ATTACKER_OBSERVATION_KEYS,
    ))
    return defender_dim, attacker_dim


def test_round_boundary_resume_is_bit_exact(tmp_path) -> None:
    defender_dim, attacker_dim = _dimensions()
    config = AttackPolicyPretrainingConfig(
        fl_rounds=6, batch_size=2, learning_starts=2,
        train_freq=1, gradient_steps=1, replay_capacity=32,
    )
    trainer = AttackPolicyPretrainer(config)
    common = dict(
        label='rl-clipmed', origin='pretrained-against-clipmed',
        aggregator=fixed_pretraining_aggregator('clipmed', clip_radius=1.0),
        replay_seed=44,
    )
    uninterrupted = trainer.train(
        env=make_deterministic_paper_env(seed=31, horizon=6),
        defender=_agent(defender_dim, 'defender', 32),
        attacker=_agent(attacker_dim, 'attacker', 33),
        **common,
    )
    paused = trainer.pause(
        env=make_deterministic_paper_env(seed=31, horizon=6),
        defender=_agent(defender_dim, 'defender', 32),
        attacker=_agent(attacker_dim, 'attacker', 33),
        stop_after_round=3,
        **common,
    )
    path = tmp_path / 'rl-clipmed-round-3.pt'
    save_attack_pretraining_checkpoint(path, paused)
    restored = load_attack_pretraining_checkpoint(path)
    resumed = trainer.resume(
        checkpoint=restored,
        env=make_deterministic_paper_env(seed=999, horizon=6),
        defender=_agent(defender_dim, 'defender', 32),
        attacker=_agent(attacker_dim, 'attacker', 999),
        aggregator=fixed_pretraining_aggregator('clipmed', clip_radius=1.0),
    )

    uninterrupted_agent = _agent(attacker_dim, 'attacker', 1001)
    uninterrupted_agent.restore(uninterrupted.policy)
    resumed_agent = _agent(attacker_dim, 'attacker', 1002)
    resumed_agent.restore(resumed.policy)

    assert paused.round_state.round_index == 3
    assert resumed_agent.fingerprint() == uninterrupted_agent.fingerprint()
    assert resumed.policy.update_count == uninterrupted.policy.update_count
    assert resumed.update_stats == uninterrupted.update_stats
    assert resumed.replay_transition_count == uninterrupted.replay_transition_count == 6
