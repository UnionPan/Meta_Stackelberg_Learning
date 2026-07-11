"""Offline research orchestration over canonical FL episodes."""

from meta_stackelberg.experiments.defense_response_surface import (
    ClipControllabilityGate,
    ClipResponsePoint,
    ClipResponseSurface,
    RawClipObservation,
    evaluate_clip_response_surface,
)
from meta_stackelberg.experiments.defense_response_matrix import (
    DefenseGridPoint,
    DefenseResponseMatrix,
    DefenseResponsePoint,
    E2GateResult,
    MatrixGateThresholds,
    RawDefenseObservation,
    TaskMatrixGateResult,
    evaluate_defense_response_matrix,
    evaluate_e2_gate,
    evaluate_task_matrix_gate,
)

__all__ = [
    'ClipControllabilityGate',
    'ClipResponsePoint',
    'ClipResponseSurface',
    'DefenseGridPoint',
    'DefenseResponseMatrix',
    'DefenseResponsePoint',
    'E2GateResult',
    'MatrixGateThresholds',
    'RawClipObservation',
    'RawDefenseObservation',
    'TaskMatrixGateResult',
    'evaluate_clip_response_surface',
    'evaluate_defense_response_matrix',
    'evaluate_e2_gate',
    'evaluate_task_matrix_gate',
]
