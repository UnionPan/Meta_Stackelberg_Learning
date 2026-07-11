from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.scaled_evidence import (
    run_deterministic_scaled_evidence,
)
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds


def test_single_entrypoint_runs_training_and_scientific_evidence() -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=2, N_A=2, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=4, learning_starts=4, hidden_sizes=(8,),
        replay_capacity=512,
    )
    result = run_deterministic_scaled_evidence(
        config=config,
        thresholds=ScientificGateThresholds(
            0.001, 0.001, 0.001, 0.001, 0.1, 0.001,
            attacker_plateau_gap=0.001,
        ),
        query_seeds=(101, 102),
        training_support_seed=2000,
        scientific_support_seeds=tuple(range(5000, 5200)),
        seed=7,
    )

    assert len(result.training.algorithm1.iterations) == 1
    assert len(result.training.algorithm2.iterations) == 1
    assert len(result.scientific.gate.checks) == 6
    assert result.parameter_snapshot['N_A'] == 2
    assert result.parameter_snapshot['H'] == 2
    assert result.query_seeds == (101, 102)
