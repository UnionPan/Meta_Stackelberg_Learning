"""Runtime state, timing, and summary helpers for sandbox FL execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - lightweight environments
    DataLoader = Any


Weights = List[np.ndarray]


class RuntimeTimer:
    """Small monotonic timer used for round and experiment durations."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @classmethod
    def start(cls) -> "RuntimeTimer":
        return cls()

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start


class RoundTimer(RuntimeTimer):
    """Timer for one federated-learning round."""


class ExperimentTimer(RuntimeTimer):
    """Timer for a full sandbox experiment."""


@dataclass
class RoundContext:
    """Minimum data needed to execute an attacker for one FL round."""

    round_idx: int
    old_weights: Weights
    benign_weights: List[Weights]
    selected_attacker_ids: List[int]
    model: Optional[Any] = None
    device: Optional[Any] = None
    fl_config: Optional[Any] = None
    defense_type: str = "fedavg"
    lr: float = 0.0
    server_lr: float = 0.0
    local_epochs: int = 1
    attacker_train_iter: Optional[DataLoader] = None
    all_attacker_train_iter: Optional[DataLoader] = None
    selected_attacker_train_loaders: Optional[Dict[int, DataLoader]] = None
    global_poisoned_train_loader: Optional[DataLoader] = None
    sub_trigger_train_loaders: Optional[List[DataLoader]] = None
    poisoned_train_iters: Optional[Dict[str, Any]] = None
    attacker_action: Optional[np.ndarray] = None
    trusted_reference_weights: Optional[Weights] = None


@dataclass
class ClientRoundMetrics:
    """Per-client training and update metrics for one round."""

    round_idx: int
    client_id: int
    selected: bool
    is_attacker: bool
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    update_norm: Optional[float] = None


@dataclass
class RoundSummary:
    """Serializable summary of one executed FL round."""

    round_idx: int
    attack_name: str
    defense_name: str
    sampled_clients: List[int]
    benign_clients: List[int]
    selected_attackers: List[int]
    clean_loss: float
    clean_acc: float
    backdoor_acc: float
    round_seconds: float
    client_metrics: List[ClientRoundMetrics] = field(default_factory=list)
    benign_update_norms: List[float] = field(default_factory=list)
    malicious_update_norms: List[float] = field(default_factory=list)
    malicious_cosines_to_benign: List[float] = field(default_factory=list)


@dataclass
class RoundRuntimeState:
    """Mutable state that is built and filled while one round executes."""

    round_idx: int
    sampled_clients: List[int]
    benign_clients: List[int]
    selected_attackers: List[int]
    client_metrics: Dict[int, ClientRoundMetrics]

    @classmethod
    def from_selection(
        cls,
        *,
        round_idx: int,
        sampled_clients: List[int],
        attacker_ids: Iterable[int],
        num_clients: int,
    ) -> "RoundRuntimeState":
        attacker_set = set(attacker_ids)
        selected_attackers = [client_id for client_id in sampled_clients if client_id in attacker_set]
        benign_clients = [client_id for client_id in sampled_clients if client_id not in attacker_set]
        client_metrics = {
            client_id: ClientRoundMetrics(
                round_idx=round_idx,
                client_id=client_id,
                selected=client_id in sampled_clients,
                is_attacker=client_id in attacker_set,
            )
            for client_id in range(num_clients)
        }
        return cls(
            round_idx=round_idx,
            sampled_clients=sampled_clients,
            benign_clients=benign_clients,
            selected_attackers=selected_attackers,
            client_metrics=client_metrics,
        )

    def record_client_training(
        self,
        client_id: int,
        *,
        train_loss: float,
        train_acc: float,
        update_norm: float,
    ) -> None:
        metrics = self.client_metrics[client_id]
        metrics.train_loss = train_loss
        metrics.train_acc = train_acc
        metrics.update_norm = update_norm

    def client_metrics_list(self) -> List[ClientRoundMetrics]:
        return [self.client_metrics[client_id] for client_id in sorted(self.client_metrics)]


@dataclass
class RoundUpdateStats:
    """Aggregated update norms and malicious-to-benign cosine stats."""

    benign_update_norms: List[float]
    malicious_update_norms: List[float]
    malicious_cosines_to_benign: List[float]


def build_round_context(
    *,
    round_idx: int,
    old_weights: Weights,
    benign_weights: List[Weights],
    selected_attacker_ids: List[int],
    model: Optional[Any],
    device: Optional[Any],
    fl_config: Optional[Any],
    defense_type: str,
    lr: float,
    local_epochs: int,
    attacker_train_iter: Optional[DataLoader],
    all_attacker_train_iter: Optional[DataLoader],
    selected_attacker_train_loaders: Optional[Dict[int, DataLoader]],
    global_poisoned_train_loader: Optional[DataLoader],
    sub_trigger_train_loaders: Optional[List[DataLoader]],
    poisoned_train_iters: Optional[Dict[str, Any]],
    attacker_action: np.ndarray,
    trusted_reference_weights: Optional[Weights],
    server_lr: float = 1.0,
) -> RoundContext:
    """Build the attacker-facing round context in one place."""

    return RoundContext(
        round_idx=round_idx,
        old_weights=old_weights,
        benign_weights=benign_weights,
        selected_attacker_ids=selected_attacker_ids,
        model=model,
        device=device,
        fl_config=fl_config,
        defense_type=defense_type,
        lr=lr,
        server_lr=server_lr,
        local_epochs=local_epochs,
        attacker_train_iter=attacker_train_iter,
        all_attacker_train_iter=all_attacker_train_iter,
        selected_attacker_train_loaders=selected_attacker_train_loaders,
        global_poisoned_train_loader=global_poisoned_train_loader,
        sub_trigger_train_loaders=sub_trigger_train_loaders,
        poisoned_train_iters=poisoned_train_iters,
        attacker_action=attacker_action,
        trusted_reference_weights=trusted_reference_weights,
    )


def summarize_round_updates(
    old_weights: Weights,
    benign_weights: List[Weights],
    malicious_weights: List[Weights],
) -> RoundUpdateStats:
    """Calculate round-level update stats in a single runtime helper."""

    from .metrics import summarize_norms, update_cosine_to_benign_mean

    malicious_cosines = [
        update_cosine_to_benign_mean(old_weights, weights, benign_weights)
        for weights in malicious_weights
    ] if benign_weights and malicious_weights else []
    return RoundUpdateStats(
        benign_update_norms=summarize_norms(old_weights, benign_weights),
        malicious_update_norms=summarize_norms(old_weights, malicious_weights),
        malicious_cosines_to_benign=malicious_cosines,
    )


def build_round_summary(
    *,
    state: RoundRuntimeState,
    attack_name: str,
    defense_name: str,
    clean_loss: float,
    clean_acc: float,
    backdoor_acc: float,
    round_seconds: float,
    update_stats: RoundUpdateStats,
) -> RoundSummary:
    """Convert completed runtime state into the public round summary."""

    return RoundSummary(
        round_idx=state.round_idx,
        attack_name=attack_name,
        defense_name=defense_name,
        sampled_clients=state.sampled_clients,
        benign_clients=state.benign_clients,
        selected_attackers=state.selected_attackers,
        clean_loss=clean_loss,
        clean_acc=clean_acc,
        backdoor_acc=backdoor_acc,
        round_seconds=round_seconds,
        client_metrics=state.client_metrics_list(),
        benign_update_norms=update_stats.benign_update_norms,
        malicious_update_norms=update_stats.malicious_update_norms,
        malicious_cosines_to_benign=update_stats.malicious_cosines_to_benign,
    )


def summaries_to_dict(summaries: List[RoundSummary]) -> Dict[str, List[float]]:
    """Flatten summaries into round-wise scalar series."""

    backdoor_acc = [summary.backdoor_acc for summary in summaries]
    return {
        "clean_loss": [summary.clean_loss for summary in summaries],
        "clean_acc": [summary.clean_acc for summary in summaries],
        "backdoor_acc": backdoor_acc,
        "asr": backdoor_acc,
        "round_seconds": [summary.round_seconds for summary in summaries],
        "defense_name": [summary.defense_name for summary in summaries],
        "mean_benign_norm": [
            float(np.nanmean(summary.benign_update_norms)) if summary.benign_update_norms else 0.0
            for summary in summaries
        ],
        "mean_malicious_norm": [
            float(np.nanmean(summary.malicious_update_norms)) if summary.malicious_update_norms else 0.0
            for summary in summaries
        ],
        "mean_malicious_cosine": [
            float(np.nanmean(summary.malicious_cosines_to_benign))
            if summary.malicious_cosines_to_benign else 0.0
            for summary in summaries
        ],
    }


def client_metrics_to_rows(summaries: List[RoundSummary]) -> List[Dict[str, object]]:
    """Flatten per-client metrics into a table-friendly row format."""

    rows: List[Dict[str, object]] = []
    for summary in summaries:
        for metric in summary.client_metrics:
            rows.append(
                {
                    "round_idx": metric.round_idx,
                    "client_id": metric.client_id,
                    "selected": metric.selected,
                    "is_attacker": metric.is_attacker,
                    "train_loss": metric.train_loss,
                    "train_acc": metric.train_acc,
                    "update_norm": metric.update_norm,
                }
            )
    return rows
