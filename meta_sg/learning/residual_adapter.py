"""Context-conditioned residual adapter for Meta-SG few-shot defense."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ATTACK_NAMES = ("clean", "ipm", "lmp", "rl", "bfl", "dba", "rl_backdoor")
OBJECTIVES = ("targeted", "untargeted")
BASE_FEATURES = (
    "support_score_mean",
    "support_clean_mean",
    "support_backdoor_mean",
    "support_reward_mean",
    "support_score_slope",
    "support_clean_slope",
    "support_backdoor_slope",
    "support_reward_slope",
    "direction_norm",
)


class ResidualAdapterNet(nn.Module):
    """Small MLP that maps support/query context features to a raw action residual."""

    def __init__(self, *, input_dim: int, act_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        return self.output(x)


@dataclass
class ResidualAdapter:
    model: ResidualAdapterNet
    feature_schema: list[str]
    act_dim: int
    bound: float


def encode_residual_features(
    support_summary: dict,
    *,
    attack_name: str,
    attack_objective: str,
    act_dim: int,
    feature_schema: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    schema = feature_schema or residual_feature_schema(act_dim)
    values = _feature_value_map(
        support_summary,
        attack_name=str(attack_name),
        attack_objective=str(attack_objective),
        act_dim=int(act_dim),
    )
    features = np.asarray([float(values.get(name, 0.0)) for name in schema], dtype=np.float32)
    features[~np.isfinite(features)] = 0.0
    return features, list(schema)


def residual_feature_schema(act_dim: int) -> list[str]:
    schema = list(BASE_FEATURES)
    schema.extend(f"direction_{idx}" for idx in range(int(act_dim)))
    schema.extend(f"normalized_direction_{idx}" for idx in range(int(act_dim)))
    schema.extend(f"objective_{name}" for name in OBJECTIVES)
    schema.extend(f"attack_{name}" for name in ATTACK_NAMES)
    return schema


def fit_residual_adapter(
    samples: list[dict],
    *,
    act_dim: int,
    bound: float,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> ResidualAdapter:
    if not samples:
        raise ValueError("fit_residual_adapter requires at least one sample")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    schema = residual_feature_schema(int(act_dim))
    x = []
    y = []
    for sample in samples:
        features, _ = encode_residual_features(
            sample.get("support_summary", {}),
            attack_name=str(sample.get("attack_name", "")),
            attack_objective=str(sample.get("attack_objective", "")),
            act_dim=int(act_dim),
            feature_schema=schema,
        )
        target = np.asarray(sample.get("selected_offset", np.zeros(int(act_dim))), dtype=np.float32)
        if target.shape[0] != int(act_dim):
            padded = np.zeros(int(act_dim), dtype=np.float32)
            usable = min(len(target), int(act_dim))
            padded[:usable] = target[:usable]
            target = padded
        target = np.clip(target, -abs(float(bound)), abs(float(bound))).astype(np.float32)
        x.append(features)
        y.append(target)
    features_t = torch.as_tensor(np.stack(x), dtype=torch.float32)
    targets_t = torch.as_tensor(np.stack(y), dtype=torch.float32)
    model = ResidualAdapterNet(input_dim=len(schema), act_dim=int(act_dim), hidden_dim=int(hidden_dim))
    optim = torch.optim.Adam(model.parameters(), lr=float(lr))
    for _ in range(max(1, int(epochs))):
        pred = torch.tanh(model(features_t)) * abs(float(bound))
        loss = F.mse_loss(pred, targets_t)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return ResidualAdapter(model=model, feature_schema=schema, act_dim=int(act_dim), bound=abs(float(bound)))


@torch.no_grad()
def predict_residual_offset(adapter: ResidualAdapter, sample: dict, *, bound: float | None = None) -> np.ndarray:
    limit = abs(float(adapter.bound if bound is None else bound))
    features, _ = encode_residual_features(
        sample.get("support_summary", {}),
        attack_name=str(sample.get("attack_name", "")),
        attack_objective=str(sample.get("attack_objective", "")),
        act_dim=int(adapter.act_dim),
        feature_schema=adapter.feature_schema,
    )
    model = adapter.model
    model.eval()
    raw = model(torch.as_tensor(features[None, :], dtype=torch.float32))[0]
    offset = (torch.tanh(raw) * limit).detach().cpu().numpy().astype(np.float32)
    return np.clip(offset, -limit, limit).astype(np.float32)


def save_residual_adapter(adapter: ResidualAdapter, path: str | Path, *, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "feature_schema": list(adapter.feature_schema),
            "act_dim": int(adapter.act_dim),
            "bound": float(adapter.bound),
            "hidden_dim": int(adapter.model.hidden_dim),
            "metrics": dict(metrics or {}),
        },
        path,
    )


def load_residual_adapter(path: str | Path) -> tuple[ResidualAdapter, dict]:
    ckpt = torch.load(Path(path), map_location="cpu")
    schema = list(ckpt["feature_schema"])
    act_dim = int(ckpt["act_dim"])
    model = ResidualAdapterNet(
        input_dim=len(schema),
        act_dim=act_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 32)),
    )
    model.load_state_dict(ckpt["state_dict"])
    adapter = ResidualAdapter(
        model=model,
        feature_schema=schema,
        act_dim=act_dim,
        bound=float(ckpt["bound"]),
    )
    return adapter, dict(ckpt.get("metrics", {}))


def _feature_value_map(
    support_summary: dict,
    *,
    attack_name: str,
    attack_objective: str,
    act_dim: int,
) -> dict[str, float]:
    values = {name: float(support_summary.get(name, 0.0)) for name in BASE_FEATURES}
    direction = _padded_vector(support_summary.get("direction", []), act_dim)
    normalized = _padded_vector(support_summary.get("normalized_direction", []), act_dim)
    values.update({f"direction_{idx}": float(direction[idx]) for idx in range(int(act_dim))})
    values.update({f"normalized_direction_{idx}": float(normalized[idx]) for idx in range(int(act_dim))})
    values.update({f"objective_{name}": 1.0 if str(attack_objective) == name else 0.0 for name in OBJECTIVES})
    values.update({f"attack_{name}": 1.0 if str(attack_name) == name else 0.0 for name in ATTACK_NAMES})
    return values


def _padded_vector(values: Iterable[float], act_dim: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    out = np.zeros(int(act_dim), dtype=np.float32)
    usable = min(len(arr), int(act_dim))
    if usable:
        out[:usable] = arr[:usable]
    out[~np.isfinite(out)] = 0.0
    return out
