from types import SimpleNamespace

import numpy as np
import pytest

from fl_sandbox.federation.runner import MinimalFLRunner
from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, MetaSGSandboxAttack
from meta_sg.strategies.attacks.fixed import IPMAttack
from meta_sg.strategies.types import AttackDecision


def test_meta_sg_sandbox_attack_wraps_strategy_execute():
    old_weights = [np.asarray([1.0, 2.0], dtype=np.float32)]
    benign_weights = [[np.asarray([1.1, 2.1], dtype=np.float32)]]
    strategy = IPMAttack(scaling=1.0)
    decision = AttackDecision.from_raw(np.asarray([0.0, 0.0], dtype=np.float32))
    wrapped = MetaSGSandboxAttack(strategy, decision)
    ctx = SimpleNamespace(
        old_weights=old_weights,
        benign_weights=benign_weights,
        selected_attacker_ids=[0, 1],
    )

    malicious = wrapped.execute(ctx)

    assert len(malicious) == 2
    assert not np.allclose(malicious[0][0], old_weights[0])


def test_fl_sandbox_adapter_evaluate_model_delegates_to_runner():
    adapter = object.__new__(FLSandboxCoordinatorAdapter)
    adapter.runner = SimpleNamespace(
        evaluate_model=lambda model, weights: {
            "clean_loss": 1.25,
            "clean_acc": 0.5,
            "backdoor_acc": 0.75,
        }
    )

    metrics = adapter.evaluate_model("model", ["weights"])

    assert metrics == {
        "clean_loss": 1.25,
        "clean_acc": 0.5,
        "backdoor_acc": 0.75,
    }


def test_fl_sandbox_adapter_reset_reseeds_existing_runner_without_reloading_datasets(monkeypatch):
    def fail_if_reconstructed(config):
        raise AssertionError("reset should not reconstruct MinimalFLRunner")

    monkeypatch.setattr(
        "meta_sg.simulation.fl_sandbox_adapter.MinimalFLRunner",
        fail_if_reconstructed,
    )

    class FakeRunner:
        def __init__(self):
            self.reset_calls = 0
            self.current_weights = [np.asarray([1.0, 2.0], dtype=np.float32)]

        def reset_model(self):
            self.reset_calls += 1

    adapter = object.__new__(FLSandboxCoordinatorAdapter)
    adapter.config = SimpleNamespace(runtime=SimpleNamespace(seed=42))
    adapter.runner = FakeRunner()
    adapter._round_idx = 3
    adapter._last_summary = "cached"

    initial = adapter.reset(seed=7)

    assert adapter.config.runtime.seed == 7
    assert adapter.runner.reset_calls == 1
    assert adapter._round_idx == 0
    assert adapter._last_summary is None
    assert np.allclose(initial.weights[0], [1.0, 2.0])


def test_minimal_fl_runner_preserves_zero_server_lr_defense_decision():
    runner = object.__new__(MinimalFLRunner)
    runner.defender = SimpleNamespace()

    defender = runner._round_defender(
        SimpleNamespace(norm_bound_alpha=3.0, trimmed_mean_beta=0.2, server_lr=0.0)
    )

    assert defender.server_lr == pytest.approx(0.0)
