from types import SimpleNamespace

import numpy as np

from meta_sg.simulation.fl_sandbox_adapter import MetaSGSandboxAttack
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
