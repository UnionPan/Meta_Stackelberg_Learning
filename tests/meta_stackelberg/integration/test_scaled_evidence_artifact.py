import json

from meta_stackelberg.agents.td3.config import PaperMetaSGConfig
from meta_stackelberg.experiments.scaled_artifact import (
    load_scaled_evidence_snapshots,
    save_scaled_evidence_artifact,
)
from meta_stackelberg.experiments.scaled_evidence import (
    run_deterministic_scaled_evidence,
)
from meta_stackelberg.experiments.scientific_gate import ScientificGateThresholds


def test_scaled_artifact_persists_snapshots_manifest_seeds_and_gate(tmp_path) -> None:
    config = PaperMetaSGConfig().scaled(
        T=1, K=1, H=2, l=1, N_A=1, N_D=1,
        workers=4, untargeted_attackers=2, sample_size=4,
        td3_batch_size=4, learning_starts=4, hidden_sizes=(8,),
        replay_capacity=128,
    )
    result = run_deterministic_scaled_evidence(
        config=config,
        thresholds=ScientificGateThresholds(
            0.001, 0.001, 0.001, 0.001, 0.1, 0.001,
            attacker_plateau_gap=0.001,
        ),
        query_seeds=(101, 102),
        training_support_seed=1000,
        scientific_support_seeds=tuple(range(2000, 2200)),
        seed=9,
    )

    artifact = save_scaled_evidence_artifact(tmp_path / 'evidence', result)
    payload = load_scaled_evidence_snapshots(artifact.snapshot_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding='utf-8'))

    assert artifact.manifest_path.is_file()
    assert artifact.snapshot_path.is_file()
    assert payload['algorithm1_defender'].role == 'defender'
    assert payload['algorithm2_defender'].role == 'defender'
    assert set(payload['algorithm1_attackers']) == {'rl-0'}
    assert manifest['parameters']['N_A'] == 1
    assert manifest['query_seeds'] == [101, 102]
    assert len(manifest['scientific']['checks']) == 6
    assert manifest['fingerprints']['algorithm1_defender'] == (
        result.training.algorithm1_defender.fingerprint()
    )
