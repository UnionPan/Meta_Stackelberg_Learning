import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv
from meta_sg.learning.collector import TrajectoryCollector
from meta_sg.learning.meta_sg_trainer import MetaSGTrainer
from meta_sg.learning.policies import ConstantActionPolicy
from meta_sg.learning.replay_buffer import ReplayBuffer
from meta_sg.learning.task_runner import AttackTaskRunner
from meta_sg.learning.config import MetaSGConfig, TD3Config
from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.learning.td3 import TD3Agent
from meta_sg.simulation.stub import StubCoordinator
from meta_sg.strategies.attacks.fixed import IPMAttack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import ATTACK_DOMAIN
from fl_sandbox.aggregators.rules import PaperActionDefender
from fl_sandbox.federation.runner import SandboxConfig


class RecordingPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.noise_values = []

    def get_action(self, obs, **kwargs):
        self.noise_values.append(kwargs.get("noise"))
        return self.action.copy()


def make_env(horizon=3):
    return BSMGEnv(
        coordinator=StubCoordinator(num_clients=5, num_attackers=1),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=horizon),
    )


def test_obs_dim_does_not_reset_or_mutate_coordinator():
    coord = StubCoordinator(num_clients=5, num_attackers=1, seed=123)
    env = BSMGEnv(
        coordinator=coord,
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=3),
    )

    assert coord.snapshot().round_idx == 0
    dim = env.obs_dim

    assert dim > 0
    assert coord.snapshot().round_idx == 0
    assert len(coord.current_weights) == 0


def test_collector_passes_exploration_noise_to_policies():
    env = make_env(horizon=2)
    defender = RecordingPolicy([0.0, 0.0, 0.0])
    attacker = RecordingPolicy([0.0, 0.0, 0.0])
    def_buffer = ReplayBuffer(20, env.obs_dim, env.act_dim)
    atk_buffer = ReplayBuffer(20, env.obs_dim, env.act_dim)

    collector = TrajectoryCollector(
        env=env,
        defender=defender,
        attacker=attacker,
        defender_buffer=def_buffer,
        attacker_buffer=atk_buffer,
        exploration_noise=0.37,
    )
    collector.collect(horizon=2, seed=1)

    assert defender.noise_values == [0.37, 0.37]
    assert attacker.noise_values == [0.37, 0.37]


def test_collector_can_skip_attacker_buffer_for_fixed_attacks():
    env = make_env(horizon=2)
    policy = ConstantActionPolicy(act_dim=env.act_dim)
    def_buffer = ReplayBuffer(20, env.obs_dim, env.act_dim)
    atk_buffer = ReplayBuffer(20, env.obs_dim, env.act_dim)

    collector = TrajectoryCollector(
        env=env,
        defender=policy,
        attacker=policy,
        defender_buffer=def_buffer,
        attacker_buffer=atk_buffer,
        store_attacker=False,
    )
    collector.collect(horizon=2, seed=1)

    assert len(def_buffer) == 2
    assert len(atk_buffer) == 0


def test_reptile_update_uses_mean_delta_once():
    trainer = MetaSGTrainer(
        coordinator_factory=lambda: StubCoordinator(num_clients=5, num_attackers=1),
        attack_domain=[ATTACK_DOMAIN["ipm"]],
        meta_config=MetaSGConfig(T=1, K=1, H_mnist=1, l=1, warmup_steps=0),
        td3_config=TD3Config(hidden_dim=8, batch_size=4, buffer_capacity=20),
        obs_dim=10,
        act_dim=3,
    )
    meta_params = trainer.defender.get_params()
    adapted = []
    for offset in (2.0, 4.0):
        adapted.append({key: value + offset for key, value in meta_params.items()})

    trainer._reptile_update(adapted, step_size=0.5)
    updated = trainer.defender.get_params()

    for key, old_value in meta_params.items():
        np.testing.assert_allclose(
            updated[key].detach().cpu().numpy(),
            (old_value + 1.5).detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_task_runner_keeps_fixed_attack_buffer_empty():
    obs_dim = make_env(horizon=2).obs_dim
    act_dim = 3
    meta_cfg = MetaSGConfig(T=1, K=1, H_mnist=2, l=1, warmup_steps=0)
    td3_cfg = TD3Config(hidden_dim=8, batch_size=4, buffer_capacity=20)
    attacker_agents = {"ipm": TD3Agent(obs_dim, act_dim, td3_cfg)}
    attacker_buffers = {"ipm": ReplayBuffer(20, obs_dim, act_dim)}
    best_response = AttackerBestResponse(attacker_agents, attacker_buffers, n_a=1)
    runner = AttackTaskRunner(
        coordinator_factory=lambda: StubCoordinator(num_clients=5, num_attackers=1),
        td3_config=td3_cfg,
        meta_config=meta_cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        attacker_agents=attacker_agents,
        attacker_buffers=attacker_buffers,
        best_response=best_response,
    )
    meta_defender = TD3Agent(obs_dim, act_dim, td3_cfg)

    result = runner.run(ATTACK_DOMAIN["ipm"], meta_defender, seed_base=0)

    assert result.trajectories_collected == 1
    assert result.transitions_collected == 2
    assert result.diagnostics["attacker_buffer_size"] == 0
    assert result.diagnostics["buffer_size"] == 2


def test_fl_sandbox_paper_defender_uses_alpha_beta():
    old = [np.zeros(2, dtype=np.float32)]
    client_weights = [
        [np.array([1.0, 1.0], dtype=np.float32)],
        [np.array([2.0, 2.0], dtype=np.float32)],
        [np.array([100.0, 100.0], dtype=np.float32)],
    ]

    loose = PaperActionDefender(norm_bound_alpha=500.0, trimmed_mean_beta=0.0)
    tight = PaperActionDefender(norm_bound_alpha=1.0, trimmed_mean_beta=0.0)

    loose_result = loose.aggregate(old, client_weights)[0]
    tight_result = tight.aggregate(old, client_weights)[0]

    assert np.linalg.norm(tight_result) < np.linalg.norm(loose_result)


def test_sandbox_config_exposes_small_experiment_limits():
    config = SandboxConfig(max_client_samples_per_client=4, max_eval_samples=8)

    assert config.max_client_samples_per_client == 4
    assert config.max_eval_samples == 8
