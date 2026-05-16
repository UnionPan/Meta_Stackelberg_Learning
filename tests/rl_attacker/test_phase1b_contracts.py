import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.attacks.rl_attacker.attack import RLAttack
from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.diagnostics import RLSim2RealDiagnostics, deploy_guard_allows
from fl_sandbox.attacks.rl_attacker.krum_projection import (
    fast_legacy_krum_surrogate_update,
    project_krum_malicious_update,
    simulate_krum_benign_surrogates,
)
from fl_sandbox.attacks.rl_attacker.observation import (
    build_legacy_clipped_median_observation,
    build_legacy_scaleaware_observation,
    build_observation_from_state,
)
from fl_sandbox.attacks.rl_attacker.proxy import GradientDistributionLearner, ProxyDatasetBuffer
from fl_sandbox.attacks.rl_attacker.simulator.fl_dynamics import legacy_craft_att
from fl_sandbox.attacks.rl_attacker.simulator import SimulatedFLEnv
from fl_sandbox.attacks.rl_attacker.trainer import build_trainer
from fl_sandbox.defenders import AggregationDefender
from fl_sandbox.core.fl_runner import SandboxConfig
from src.models.cnn import MNISTClassifier


def _weights(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def _proxy_buffer(config=None):
    config = config or RLAttackerConfig(seed_samples=8, reconstruction_batch_size=4)
    images = torch.randn(12, 1, 28, 28)
    labels = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    learner = GradientDistributionLearner(config)
    learner.initialize_from_loader(loader, torch.device("cpu"))
    return learner.buffer


def test_public_package_has_no_legacy_td3_exports():
    package = importlib.import_module("fl_sandbox.attacks.rl_attacker")

    assert "ReplayBuffer" not in package.__all__
    assert "TD3Agent" not in package.__all__
    assert "PaperRLAttacker" not in package.__all__


def test_lazy_public_attack_import_does_not_import_tianshou():
    code = (
        "import sys\n"
        "from fl_sandbox.attacks import RLAttack\n"
        "assert RLAttack.__name__ == 'RLAttack'\n"
        "assert 'tianshou' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": os.getcwd()}

    result = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(), env=env, text=True)

    assert result.returncode == 0


def test_action_decoder_is_defense_aware():
    config = RLAttackerConfig()

    krum = decode_action(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), "krum", config)
    clipped = decode_action(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), "clipped_median", config)

    assert krum.gamma_scale == config.krum_gamma_center
    assert clipped.gamma_scale == config.clipmed_gamma_center
    assert krum.local_steps > 0
    assert clipped.local_steps > 0


def test_legacy_clipped_median_action_contract():
    config = RLAttackerConfig(attacker_semantics="legacy_clipped_median")

    decoded = decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "clipped_median", config)

    assert config.action_dim("clipped_median") == 2
    assert decoded.gamma_scale == 15.0
    assert decoded.local_steps == 25
    assert decoded.lambda_stealth == 0.0
    assert decoded.local_search_lr == 0.01


def test_strict_legacy_clipped_median_uses_same_action_contract():
    config = RLAttackerConfig(attacker_semantics="legacy_clipped_median_strict")

    decoded = decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "clipped_median", config)

    assert config.action_dim("clipped_median") == 2
    assert decoded.gamma_scale == 15.0
    assert decoded.local_steps == 25
    assert decoded.lambda_stealth == 0.0
    assert decoded.local_search_lr == 0.01


def test_scaleaware_clipped_median_keeps_original_action_and_strict_schedule():
    config = RLAttackerConfig(attacker_semantics="legacy_clipped_median_scaleaware")

    decoded = decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "clipped_median", config)

    assert config.action_dim("clipped_median") == 2
    assert config.uses_strict_reproduction() is True
    assert config.uses_scaleaware_legacy_observation() is True
    assert decoded.gamma_scale == 15.0
    assert decoded.local_steps == 25
    assert decoded.lambda_stealth == 0.0
    assert decoded.local_search_lr == 0.01


def test_legacy_krum_strict_action_contract_matches_original_repo():
    config = RLAttackerConfig(attacker_semantics="legacy_krum_strict")

    decoded = decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "krum", config)
    low = decode_action(np.asarray([-1.0, -1.0], dtype=np.float32), "krum", config)
    high = decode_action(np.asarray([1.0, 1.0], dtype=np.float32), "krum", config)

    assert config.action_dim("krum") == 2
    assert decoded.gamma_scale == 5.0
    assert decoded.local_steps == 11
    assert decoded.lambda_stealth == 0.0
    assert decoded.local_search_lr == 0.01
    assert decoded.template_name == "legacy_krum"
    assert low.gamma_scale == 0.1
    assert low.local_steps == 1
    assert high.gamma_scale == 9.9
    assert high.local_steps == 21


def test_legacy_krum_geometry_keeps_paper_action_contract():
    config = RLAttackerConfig(attacker_semantics="legacy_krum_geometry")

    decoded = decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "krum", config)

    assert config.action_dim("krum") == 2
    assert config.uses_legacy_krum() is True
    assert config.uses_legacy_krum_geometry() is True
    assert config.uses_strict_reproduction() is True
    assert decoded.gamma_scale == 5.0
    assert decoded.local_steps == 11
    assert decoded.template_name == "legacy_krum"


def test_legacy_krum_geometry_projects_raw_update_into_selected_region():
    old_weights = [np.asarray([0.0, 0.0], dtype=np.float32)]
    benign_deltas = np.asarray(
        [
            [0.12, 0.0],
            [-0.12, 0.0],
            [0.0, 0.12],
            [0.0, -0.12],
            [0.10, 0.10],
            [-0.10, -0.10],
            [0.10, -0.10],
        ],
        dtype=np.float32,
    )
    benign_weights = [[delta.copy()] for delta in benign_deltas]
    raw_malicious = [np.asarray([5.0, 5.0], dtype=np.float32)]

    result = project_krum_malicious_update(
        old_weights=old_weights,
        raw_malicious_weights=raw_malicious,
        benign_weights=benign_weights,
        num_attackers=3,
        num_byzantine=20,
        max_alpha=5.0,
    )

    assert result.metrics["rl_krum_raw_selected"] == 0.0
    assert result.metrics["rl_krum_projected_selected"] == 1.0
    assert result.metrics["rl_krum_projection_alpha"] > 0.0
    assert result.metrics["rl_krum_projected_delta_norm"] < result.metrics["rl_krum_raw_delta_norm"]


def test_legacy_krum_geometry_benign_surrogate_is_deterministic_and_training_free():
    old_weights = [np.asarray([0.0, 0.0], dtype=np.float32)]
    last_update = [np.asarray([0.02, -0.01], dtype=np.float32)]

    first = simulate_krum_benign_surrogates(
        old_weights=old_weights,
        last_aggregate_update=last_update,
        benign_count=4,
        seed=17,
        round_idx=3,
    )
    second = simulate_krum_benign_surrogates(
        old_weights=old_weights,
        last_aggregate_update=last_update,
        benign_count=4,
        seed=17,
        round_idx=3,
    )

    assert len(first) == 4
    assert all(layer[0].dtype == np.float32 for layer in first)
    assert [weights[0].tolist() for weights in first] == [weights[0].tolist() for weights in second]
    assert np.linalg.norm(first[0][0] - old_weights[0]) > 0.0


def test_legacy_krum_geometry_surrogate_update_tracks_benign_cluster():
    old_weights = [np.asarray([0.0, 0.0], dtype=np.float32)]
    benign_weights = [
        [np.asarray([0.04, 0.01], dtype=np.float32)],
        [np.asarray([0.05, 0.00], dtype=np.float32)],
        [np.asarray([0.03, 0.02], dtype=np.float32)],
    ]

    crafted = fast_legacy_krum_surrogate_update(
        old_weights=old_weights,
        benign_weights=benign_weights,
        gamma_scale=5.0,
        local_steps=11,
        steps_center=11.0,
        current_num_attackers=3,
        seed=19,
        round_idx=7,
    )

    benign_center = np.mean([weights[0] - old_weights[0] for weights in benign_weights], axis=0)
    crafted_delta = crafted[0] - old_weights[0]
    assert crafted[0].dtype == np.float32
    assert 0.0 < np.linalg.norm(crafted_delta - benign_center) < 0.2
    assert float(np.dot(crafted_delta - benign_center, -benign_center)) > 0.0


def test_rl_attack_clears_action_metrics_when_no_attackers_selected():
    attack = RLAttack(config=RLAttackerConfig(attacker_semantics="legacy_krum_geometry"))
    attack._last_action_metrics = {"rl_krum_projected_selected": 1.0}
    ctx = SimpleNamespace(selected_attacker_ids=[])

    assert attack.execute(ctx) == []
    assert attack._last_action_metrics == {}


def test_legacy_clipped_median_rejects_other_defenses():
    config = RLAttackerConfig(attacker_semantics="legacy_clipped_median")

    try:
        decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "krum", config)
    except ValueError as exc:
        assert "legacy_clipped_median" in str(exc)
        assert "clipped_median" in str(exc)
    else:
        raise AssertionError("legacy_clipped_median must reject non-clipped_median defenses")


def test_legacy_krum_strict_rejects_other_defenses():
    config = RLAttackerConfig(attacker_semantics="legacy_krum_strict")

    try:
        decode_action(np.asarray([0.0, 0.0], dtype=np.float32), "clipped_median", config)
    except ValueError as exc:
        assert "legacy_krum" in str(exc)
        assert "krum" in str(exc)
    else:
        raise AssertionError("legacy_krum_strict must reject non-krum defenses")


def test_strict_reproduction_attacker_dataset_schedule_is_capped():
    config = RLAttackerConfig(
        attacker_semantics="legacy_clipped_median_strict",
        strict_reproduction_initial_samples=3,
        strict_reproduction_samples_per_epoch=2,
    )

    assert config.strict_reproduction_sample_limit(epoch=1, buffer_size=10) == 3
    assert config.strict_reproduction_sample_limit(epoch=2, buffer_size=10) == 5
    assert config.strict_reproduction_sample_limit(epoch=5, buffer_size=10) == 10


def test_policy_train_steps_per_round_overrides_episode_horizon_product():
    config = RLAttackerConfig(
        policy_train_steps_per_round=200,
        episodes_per_observation=3,
        simulator_horizon=1000,
    )

    assert config.policy_train_steps() == 200


def test_offline_rolling_checkpoint_selects_latest_policy_not_after_round(tmp_path):
    for round_idx in (50, 100, 150, 400, 1000):
        tmp_path.joinpath(f"rl_policy_round_{round_idx:06d}.pt").write_bytes(b"stub")
    config = RLAttackerConfig(
        policy_checkpoint_dir=str(tmp_path),
        policy_train_end_round=400,
    )

    assert config.policy_checkpoint_for_round(1) == ""
    assert config.policy_checkpoint_for_round(101).endswith("rl_policy_round_000100.pt")
    assert config.policy_checkpoint_for_round(149).endswith("rl_policy_round_000100.pt")
    assert config.policy_checkpoint_for_round(150).endswith("rl_policy_round_000150.pt")
    assert config.policy_checkpoint_for_round(999).endswith("rl_policy_round_000400.pt")


def test_rl_attack_reloads_offline_rolling_checkpoint_when_round_crosses_checkpoint(tmp_path):
    for round_idx in (100, 150):
        torch.save({"algorithm": {}, "diagnostics": {}}, tmp_path / f"rl_policy_round_{round_idx:06d}.pt")
    config = RLAttackerConfig(
        attacker_semantics="legacy_clipped_median_strict",
        policy_checkpoint_dir=str(tmp_path),
        policy_train_end_round=400,
        freeze_policy=True,
    )
    attack = RLAttack(config=config)

    class _FakeTrainer:
        initialized = True

        def __init__(self):
            self.loaded = []

        def load(self, path):
            self.loaded.append(path)

    attack.trainer = _FakeTrainer()
    ctx = SimpleNamespace(round_idx=149, defense_type="clipped_median")

    attack._ensure_policy_trainer(ctx, np.zeros(config.action_dim("clipped_median"), dtype=np.float32))
    ctx.round_idx = 150
    attack._ensure_policy_trainer(ctx, np.zeros(config.action_dim("clipped_median"), dtype=np.float32))

    assert [path.rsplit("/", 1)[-1] for path in attack.trainer.loaded] == [
        "rl_policy_round_000100.pt",
        "rl_policy_round_000150.pt",
    ]


def test_legacy_craft_att_moves_opposite_trained_update():
    old_weights = [np.asarray([1.0, 2.0], dtype=np.float32)]
    trained_weights = [np.asarray([0.5, 3.0], dtype=np.float32)]

    crafted = legacy_craft_att(old_weights, trained_weights, gamma_scale=2.0)

    assert np.allclose(crafted[0], np.asarray([2.0, 0.0], dtype=np.float32))


def test_legacy_clipped_median_observation_uses_normalized_tail_layers_and_attacker_count():
    weights = [
        np.asarray([99.0], dtype=np.float32),
        np.asarray([[1.0, 3.0]], dtype=np.float32),
        np.asarray([5.0], dtype=np.float32),
    ]

    obs = build_legacy_clipped_median_observation(weights, num_attackers=2, tail_layers=2)

    assert obs.dtype == np.float32
    assert np.allclose(obs, np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))


def test_scaleaware_observation_preserves_absolute_tail_scale():
    small = [
        np.asarray([99.0], dtype=np.float32),
        np.asarray([[1.0, 3.0]], dtype=np.float32),
        np.asarray([5.0], dtype=np.float32),
    ]
    large = [
        np.asarray([99.0], dtype=np.float32),
        np.asarray([[10.0, 30.0]], dtype=np.float32),
        np.asarray([50.0], dtype=np.float32),
    ]

    small_obs = build_legacy_scaleaware_observation(small, num_attackers=2, round_idx=10, total_rounds=100)
    large_obs = build_legacy_scaleaware_observation(large, num_attackers=2, round_idx=10, total_rounds=100)

    assert small_obs.dtype == np.float32
    assert large_obs.dtype == np.float32
    assert small_obs.shape == large_obs.shape
    assert not np.allclose(small_obs, large_obs)
    assert large_obs[-4] > small_obs[-4]  # log L2 scale


def test_observation_from_state_is_flat_and_attacker_aware():
    obs = build_observation_from_state({"pram": np.asarray([1.0, 2.0]), "num_attacker": 1}, max_attackers=2)

    assert obs.dtype == np.float32
    assert obs.shape == (3,)
    assert obs[-1] == 0.0


def test_proxy_buffer_tracks_reconstruction_acceptance():
    buffer = ProxyDatasetBuffer(limit=8)
    buffer.add_batch(torch.zeros(2, 1, 28, 28), torch.zeros(2, dtype=torch.long), reconstructed=True)
    buffer.reject_reconstruction(2)

    assert len(buffer) == 2
    assert buffer.reconstruction_accept_rate == 0.5


def test_simulator_env_reset_and_step_return_flat_state():
    model = MNISTClassifier()
    config = RLAttackerConfig(simulator_horizon=1, local_search_batch_size=4)
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum"),
        config=config,
        fl_config=SandboxConfig(num_clients=4, num_attackers=1, batch_size=4),
        device=torch.device("cpu"),
    )

    obs = sim.reset(_weights(model))
    obs_next, reward, done = sim.step(np.zeros(config.action_dim("krum"), dtype=np.float32))

    assert obs.ndim == 1
    assert obs_next.shape == obs.shape
    assert isinstance(reward, float)
    assert done is True


def test_legacy_krum_geometry_simulator_uses_vector_benign_surrogate():
    model = MNISTClassifier()
    config = RLAttackerConfig(
        attacker_semantics="legacy_krum_geometry",
        simulator_horizon=1,
        local_search_batch_size=4,
    )
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum", krum_attackers=3),
        config=config,
        fl_config=SandboxConfig(num_clients=10, num_attackers=3, subsample_rate=1.0, batch_size=4, krum_attackers=3),
        device=torch.device("cpu"),
    )
    weights = _weights(model)
    sim.reset(weights)
    sim.current_num_attackers = 3
    sim._sample_num_attackers = lambda require_positive=False: 3
    sim._evaluate_metrics = lambda _weights: (0.0, 0.0)
    sim.defender.aggregate = lambda old_weights, new_weights, trusted_weights=None: [layer.copy() for layer in old_weights]
    calls = {"benign": 0}

    def fake_benign_update(old_weights):
        calls["benign"] += 1
        return [layer + 0.01 for layer in old_weights]

    sim._simulate_benign_update = fake_benign_update
    sim._simulate_malicious_weight = lambda old_weights, action, benign_weights: [
        layer.copy() for layer in old_weights
    ]

    sim.step(np.zeros(config.action_dim("krum"), dtype=np.float32))

    assert calls["benign"] == 0


def test_legacy_krum_geometry_simulator_malicious_path_skips_proxy_craft(monkeypatch):
    model = MNISTClassifier()
    config = RLAttackerConfig(
        attacker_semantics="legacy_krum_geometry",
        simulator_horizon=1,
        local_search_batch_size=4,
    )
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum", krum_attackers=3),
        config=config,
        fl_config=SandboxConfig(num_clients=10, num_attackers=3, subsample_rate=1.0, batch_size=4, krum_attackers=3),
        device=torch.device("cpu"),
    )
    weights = _weights(model)
    benign_weights = [[layer + 0.01 for layer in weights] for _ in range(7)]

    def fail_craft(**_kwargs):
        raise AssertionError("legacy_krum_geometry simulator should use the fast raw update approximation")

    monkeypatch.setattr("fl_sandbox.attacks.rl_attacker.simulator.env.craft_malicious_update", fail_craft)

    crafted = sim._simulate_malicious_weight(
        weights,
        np.zeros(config.action_dim("krum"), dtype=np.float32),
        benign_weights=benign_weights,
    )

    assert len(crafted) == len(weights)


def test_legacy_krum_geometry_simulator_malicious_path_skips_full_projection():
    from fl_sandbox.attacks.rl_attacker.simulator import env as simulator_env

    model = MNISTClassifier()
    config = RLAttackerConfig(
        attacker_semantics="legacy_krum_geometry",
        simulator_horizon=1,
        local_search_batch_size=4,
    )
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum", krum_attackers=3),
        config=config,
        fl_config=SandboxConfig(num_clients=10, num_attackers=3, subsample_rate=1.0, batch_size=4, krum_attackers=3),
        device=torch.device("cpu"),
    )
    weights = _weights(model)
    benign_weights = [[layer + 0.01 for layer in weights] for _ in range(7)]

    assert not hasattr(simulator_env, "project_krum_malicious_update")

    crafted = sim._simulate_malicious_weight(
        weights,
        np.zeros(config.action_dim("krum"), dtype=np.float32),
        benign_weights=benign_weights,
    )

    assert len(crafted) == len(weights)


def test_legacy_krum_geometry_simulator_step_skips_full_krum_aggregate():
    model = MNISTClassifier()
    config = RLAttackerConfig(
        attacker_semantics="legacy_krum_geometry",
        simulator_horizon=1,
        local_search_batch_size=4,
    )
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum", krum_attackers=3),
        config=config,
        fl_config=SandboxConfig(num_clients=10, num_attackers=3, subsample_rate=1.0, batch_size=4, krum_attackers=3),
        device=torch.device("cpu"),
    )
    weights = _weights(model)
    sim.reset(weights)
    sim.current_num_attackers = 3
    sim._sample_num_attackers = lambda require_positive=False: 3
    sim._evaluate_metrics = lambda _weights: (0.0, 0.0)
    sim._simulate_malicious_weight = lambda old_weights, action, benign_weights: [
        layer + 0.01 for layer in old_weights
    ]

    def fail_aggregate(*_args, **_kwargs):
        raise AssertionError("optimized krum simulator should not run full Krum aggregate")

    sim.defender.aggregate = fail_aggregate

    sim.step(np.zeros(config.action_dim("krum"), dtype=np.float32))


def test_legacy_krum_geometry_simulator_uses_fast_surrogate_metrics():
    model = MNISTClassifier()
    config = RLAttackerConfig(
        attacker_semantics="legacy_krum_geometry",
        simulator_horizon=1,
        local_search_batch_size=4,
    )
    eval_loader = DataLoader(TensorDataset(torch.randn(8, 1, 28, 28), torch.zeros(8, dtype=torch.long)), batch_size=4)
    sim = SimulatedFLEnv(
        model_template=model,
        proxy_buffer=_proxy_buffer(config),
        defender=AggregationDefender(defense_type="krum", krum_attackers=3),
        config=config,
        fl_config=SandboxConfig(num_clients=10, num_attackers=3, subsample_rate=1.0, batch_size=4, krum_attackers=3),
        device=torch.device("cpu"),
        eval_loader=eval_loader,
    )
    proxy_calls = {"count": 0}

    def fake_eval_metrics(_weights):
        raise AssertionError("optimized krum simulator should not run eval_loader inside RL steps")

    def fake_proxy_metrics(_weights):
        raise AssertionError("optimized krum simulator should not run proxy model eval inside RL steps")

    sim._evaluate_eval_metrics = fake_eval_metrics
    sim._evaluate_proxy_metrics = fake_proxy_metrics

    loss, acc = sim._evaluate_metrics(_weights(model))

    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0
    assert proxy_calls["count"] == 0


def test_build_trainer_defaults_to_td3_and_can_build_ppo():
    assert build_trainer(RLAttackerConfig()).algorithm_name == "td3"
    assert build_trainer(RLAttackerConfig(algorithm="ppo")).algorithm_name == "ppo"


def test_deploy_guard_and_diagnostics_report_gap():
    diagnostics = RLSim2RealDiagnostics(simulated_reward=4.0, real_reward=1.5, gap=2.5)

    assert deploy_guard_allows(proxy_samples=8, sim2real_gap=2.5, min_proxy_samples=8, max_gap=3.0)
    assert not deploy_guard_allows(proxy_samples=7, sim2real_gap=2.5, min_proxy_samples=8, max_gap=3.0)
    assert diagnostics.as_dict()["rl_sim2real_gap"] == 2.5
