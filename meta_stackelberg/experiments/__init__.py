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
    MatrixGateThresholds,
    RawDefenseObservation,
    TaskMatrixGateResult,
    evaluate_defense_response_matrix,
    evaluate_task_matrix_gate,
)

__all__ = [
    'ClipControllabilityGate',
    'ClipResponsePoint',
    'ClipResponseSurface',
    'DefenseGridPoint',
    'DefenseResponseMatrix',
    'DefenseResponsePoint',
    'MatrixGateThresholds',
    'RawClipObservation',
    'RawDefenseObservation',
    'TaskMatrixGateResult',
    'evaluate_clip_response_surface',
    'evaluate_defense_response_matrix',
    'evaluate_task_matrix_gate',
]
