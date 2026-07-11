"""Stable model-tail observation used by paper-aligned policies."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from meta_stackelberg.core.model_state import ModelState


@dataclass(frozen=True)
class ModelTailObservationEncoder:
    block_names: tuple[str, str]
    parameter_indices: tuple[int, ...]
    schema_version: int = 1

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> ModelTailObservationEncoder:
        if not isinstance(model, torch.nn.Module):
            raise TypeError('model must be a torch.nn.Module')
        named_parameters = tuple(model.named_parameters())
        index_by_name = {name: index for index, (name, _) in enumerate(named_parameters)}
        blocks = []
        for module_name, module in model.named_modules():
            direct = tuple(module.named_parameters(recurse=False))
            if not direct:
                continue
            indices = tuple(
                index_by_name[
                    f'{module_name}.{parameter_name}' if module_name else parameter_name
                ]
                for parameter_name, parameter in direct
                if parameter.requires_grad
            )
            if indices:
                blocks.append((module_name, indices))
        if len(blocks) < 2:
            raise ValueError('model must contain at least two learnable blocks')
        selected = blocks[-2:]
        return cls(
            block_names=(selected[0][0], selected[1][0]),
            parameter_indices=selected[0][1] + selected[1][1],
        )

    def encode(
        self,
        state: ModelState,
        *,
        round_index: int,
        horizon: int,
    ) -> Mapping[str, np.ndarray]:
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ValueError('round_index must be a non-negative integer')
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('horizon must be a positive integer')
        if round_index > horizon:
            raise ValueError('round_index must not exceed horizon')
        if max(self.parameter_indices) >= len(state.tensors):
            raise ValueError('model state does not match observation schema')
        vector = np.concatenate([
            state.tensors[index].reshape(-1).astype(np.float64, copy=False)
            for index in self.parameter_indices
        ])
        norm = float(np.linalg.norm(vector))
        normalized = (vector / max(1.0, norm)).astype(np.float32)
        progress = np.array([round_index / horizon], dtype=np.float32)
        return MappingProxyType({
            'model_tail': _readonly(normalized),
            'round_progress': _readonly(progress),
        })

    def attacker_observation(
        self,
        defender_observation: Mapping[str, np.ndarray],
        *,
        malicious_count: int,
        defender_raw_action: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        if (
            isinstance(malicious_count, bool)
            or not isinstance(malicious_count, int)
            or malicious_count < 0
        ):
            raise ValueError('malicious_count must be a non-negative integer')
        raw = np.asarray(defender_raw_action)
        if raw.shape != (3,) or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError('defender_raw_action must be a floating shape-(3,) vector')
        if not np.all(np.isfinite(raw)) or np.any(raw < -1.0) or np.any(raw > 1.0):
            raise ValueError('defender_raw_action must be finite within [-1,1]')
        return MappingProxyType({
            'model_tail': _readonly(np.asarray(defender_observation['model_tail'], dtype=np.float32)),
            'round_progress': _readonly(np.asarray(defender_observation['round_progress'], dtype=np.float32)),
            'malicious_count': _readonly(np.array([malicious_count], dtype=np.float32)),
            'defender_action': _readonly(raw.astype(np.float32)),
        })


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
