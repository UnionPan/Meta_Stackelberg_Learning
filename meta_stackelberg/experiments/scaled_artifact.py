"""Atomic policy snapshots and JSON manifests for scaled Meta-SG evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

import torch

from meta_stackelberg.agents.td3.agent import TD3Snapshot
from meta_stackelberg.experiments.scaled_evidence import ScaledEvidenceResult


@dataclass(frozen=True)
class ScaledEvidenceArtifact:
    directory: Path
    manifest_path: Path
    snapshot_path: Path
    schema_version: int = 1


def save_scaled_evidence_artifact(
    directory: str | Path,
    result: ScaledEvidenceResult,
) -> ScaledEvidenceArtifact:
    if not isinstance(result, ScaledEvidenceResult):
        raise TypeError('result must be ScaledEvidenceResult')
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_path = target / 'policies.pt'
    manifest_path = target / 'manifest.json'
    attacker_snapshots = {}
    attacker_fingerprints = {}
    for task, agent in result.training.algorithm1_attackers.items():
        label = str(task)
        if label in attacker_snapshots:
            raise ValueError('task labels collide after string serialization')
        attacker_snapshots[label] = agent.snapshot()
        attacker_fingerprints[label] = agent.fingerprint()
    snapshot_payload = {
        'schema_version': 1,
        'protocol': result.protocol,
        'algorithm1_defender': result.training.algorithm1_defender.snapshot(),
        'algorithm2_defender': result.training.algorithm2_defender.snapshot(),
        'algorithm1_attackers': attacker_snapshots,
    }
    _atomic_torch_save(snapshot_path, snapshot_payload)
    scientific = result.scientific
    manifest = {
        'schema_version': 1,
        'protocol': result.protocol,
        'parameters': dict(result.parameter_snapshot),
        'query_seeds': list(result.query_seeds),
        'training': {
            'support_seeds': list(result.training.support_seeds),
            'trajectory_count': result.training.trajectory_count,
            'trajectories_per_update': result.training.trajectories_per_update,
        },
        'scientific': {
            'passed': scientific.gate.passed,
            'protocol': scientific.gate.protocol,
            'thresholds': asdict(scientific.gate.thresholds),
            'checks': [asdict(check) for check in scientific.gate.checks],
            'used_support_seeds': list(scientific.used_support_seeds),
            'query_seeds': list(scientific.query_seeds),
            'fresh_response_count': scientific.fresh_response_count,
            'specialized_oracle_label': scientific.specialized_oracle_label,
            'attacker_oracle_label': scientific.attacker_oracle_label,
            'budgets': {
                label: asdict(budget) for label, budget in scientific.budgets.items()
            },
            'adaptation_seed_blocks': {
                label: list(seeds)
                for label, seeds in scientific.adaptation_seed_blocks.items()
            },
        },
        'fingerprints': {
            'algorithm1_defender': result.training.algorithm1_defender.fingerprint(),
            'algorithm2_defender': result.training.algorithm2_defender.fingerprint(),
            'algorithm1_attackers': attacker_fingerprints,
        },
        'snapshot_file': snapshot_path.name,
    }
    _atomic_text_save(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n',
    )
    return ScaledEvidenceArtifact(target, manifest_path, snapshot_path)


def load_scaled_evidence_snapshots(path: str | Path) -> dict:
    """Load snapshots produced locally by this package; never load untrusted pickle."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('unknown scaled evidence snapshot schema')
    if not isinstance(payload.get('algorithm1_defender'), TD3Snapshot):
        raise ValueError('scaled evidence omits Algorithm 1 Defender snapshot')
    if not isinstance(payload.get('algorithm2_defender'), TD3Snapshot):
        raise ValueError('scaled evidence omits Algorithm 2 Defender snapshot')
    attackers = payload.get('algorithm1_attackers')
    if not isinstance(attackers, dict) or any(
        not isinstance(snapshot, TD3Snapshot) for snapshot in attackers.values()
    ):
        raise ValueError('scaled evidence attacker snapshots are invalid')
    return payload


def _atomic_torch_save(path: Path, payload) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text_save(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
