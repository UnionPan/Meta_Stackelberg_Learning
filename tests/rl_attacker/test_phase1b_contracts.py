import importlib
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_sandbox.attacks.rl_attacker.action_decoder import decode_action
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.diagnostics import RLSim2RealDiagnostics, deploy_guard_allows
from fl_sandbox.attacks.rl_attacker.observation import build_observation_from_state
from fl_sandbox.attacks.rl_attacker.proxy import GradientDistributionLearner, ProxyDatasetBuffer
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


def test_build_trainer_defaults_to_sac_and_can_build_td3():
    assert build_trainer(RLAttackerConfig()).algorithm_name == "sac"
    assert build_trainer(RLAttackerConfig(algorithm="td3")).algorithm_name == "td3"


def test_deploy_guard_and_diagnostics_report_gap():
    diagnostics = RLSim2RealDiagnostics(simulated_reward=4.0, real_reward=1.5, gap=2.5)

    assert deploy_guard_allows(proxy_samples=8, sim2real_gap=2.5, min_proxy_samples=8, max_gap=3.0)
    assert not deploy_guard_allows(proxy_samples=7, sim2real_gap=2.5, min_proxy_samples=8, max_gap=3.0)
    assert diagnostics.as_dict()["rl_sim2real_gap"] == 2.5
