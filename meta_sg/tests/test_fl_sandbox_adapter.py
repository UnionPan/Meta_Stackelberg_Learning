from types import SimpleNamespace

import numpy as np

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
