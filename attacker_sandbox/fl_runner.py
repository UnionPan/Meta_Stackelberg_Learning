"""Minimal federated-learning runner for attacker validation."""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn import MNISTClassifier, ResNet18
from src.utils.data_loader import DatasetSplit, get_datasets
from src.utils.fl_utils import resolve_device, set_parameters, test

from .attacks import SandboxAttack
from .context import RoundContext
from .metrics import summarize_norms, update_cosine_to_benign_mean, update_norm

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - best-effort UX fallback
    tqdm = None

@dataclass
class SandboxConfig:
    dataset: str = "mnist"
    data_dir: str = "data"
    device: str = "auto"
    seed: int = 42
    num_clients: int = 10
    num_attackers: int = 2
    subsample_rate: float = 0.5
    local_epochs: int = 1
    lr: float = 0.05
    batch_size: int = 64
    eval_batch_size: int = 2048
    num_workers: Optional[int] = None
    prefetch_factor: int = 4
    parallel_clients: int = 1


@dataclass
class RoundSummary:
    round_idx: int
    attack_name: str
    sampled_clients: List[int]
    benign_clients: List[int]
    selected_attackers: List[int]
    clean_loss: float
    clean_acc: float
    round_seconds: float
    client_metrics: List["ClientRoundMetrics"] = field(default_factory=list)
    benign_update_norms: List[float] = field(default_factory=list)
    malicious_update_norms: List[float] = field(default_factory=list)
    malicious_cosines_to_benign: List[float] = field(default_factory=list)


@dataclass
class ClientRoundMetrics:
    round_idx: int
    client_id: int
    selected: bool
    is_attacker: bool
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    update_norm: Optional[float] = None


class MinimalFLRunner:
    """A compact FL runner meant only for attacker-side validation."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.device = resolve_device(self.config.device)
        self.num_workers = self._resolve_num_workers(self.config.num_workers)
        self.loader_kwargs = self._make_loader_kwargs(shuffle=True)
        self.eval_loader_kwargs = self._make_loader_kwargs(shuffle=False)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._set_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.train_dataset, self.test_dataset = get_datasets(
            self.config.dataset,
            data_dir=self.config.data_dir,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            **self.eval_loader_kwargs,
        )

        self.client_data_idxs = self._split_data()
        self.attacker_ids = list(range(self.config.num_attackers))
        self.client_loaders = [
            DataLoader(
                DatasetSplit(self.train_dataset, list(self.client_data_idxs[client_id])),
                batch_size=self.config.batch_size,
                **self.loader_kwargs,
            )
            for client_id in range(self.config.num_clients)
        ]

        self.model = self._build_model().to(self.device)
        self.client_model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            self.client_model = self.client_model.to(memory_format=torch.channels_last)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=self.config.lr)
        self.current_weights = self._capture_weights(self.model)

    def reset_model(self) -> None:
        """Reinitialize model weights for a fresh sandbox run."""
        self._set_seed(self.config.seed)
        self.model = self._build_model().to(self.device)
        self.client_model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            self.client_model = self.client_model.to(memory_format=torch.channels_last)
        self.client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=self.config.lr)
        self.current_weights = self._capture_weights(self.model)

    def run_round(
        self,
        round_idx: int,
        attack: Optional[SandboxAttack] = None,
        evaluate: bool = True,
    ) -> RoundSummary:
        """Execute one federated round with an optional attacker."""
        round_start = time.time()
        old_weights = [weights.copy() for weights in self.current_weights]
        sampled_clients = self._sample_clients(round_idx)
        selected_attackers = [cid for cid in sampled_clients if cid in self.attacker_ids]
        benign_clients = [cid for cid in sampled_clients if cid not in self.attacker_ids]

        benign_weights = []
        client_metrics = {
            client_id: ClientRoundMetrics(
                round_idx=round_idx,
                client_id=client_id,
                selected=client_id in sampled_clients,
                is_attacker=client_id in self.attacker_ids,
            )
            for client_id in range(self.config.num_clients)
        }
        if benign_clients:
            if self.config.parallel_clients > 1:
                for cid, weights, train_loss, train_acc in self._train_clients_parallel(old_weights, benign_clients):
                    benign_weights.append(weights)
                    client_metrics[cid].train_loss = train_loss
                    client_metrics[cid].train_acc = train_acc
                    client_metrics[cid].update_norm = update_norm(old_weights, weights)
            else:
                for cid in benign_clients:
                    weights, train_loss, train_acc = self._train_client(old_weights, cid)
                    benign_weights.append(weights)
                    client_metrics[cid].train_loss = train_loss
                    client_metrics[cid].train_acc = train_acc
                    client_metrics[cid].update_norm = update_norm(old_weights, weights)
        all_weights = list(benign_weights)

        malicious_weights: List[List[np.ndarray]] = []
        attack_name = attack.name if attack is not None else "clean"
        if attack is not None and selected_attackers:
            ctx = RoundContext(
                old_weights=old_weights,
                benign_weights=benign_weights,
                selected_attacker_ids=selected_attackers,
                model=None,
                device=self.device,
                lr=self.config.lr,
                attacker_train_iter=None,
                poisoned_train_iters=None,
            )
            malicious_weights = attack.execute(ctx)
            all_weights.extend(malicious_weights)

        if all_weights:
            self.current_weights = self._average_weights(all_weights)
            set_parameters(self.model, self.current_weights)

        if evaluate:
            clean_loss, clean_acc = test(self.model, self.test_loader, device=self.device)
        else:
            clean_loss, clean_acc = float("nan"), float("nan")
        benign_norms = summarize_norms(old_weights, benign_weights)
        malicious_norms = summarize_norms(old_weights, malicious_weights)
        malicious_cosines = [
            update_cosine_to_benign_mean(old_weights, weights, benign_weights)
            for weights in malicious_weights
        ] if benign_weights and malicious_weights else []
        round_seconds = time.time() - round_start

        return RoundSummary(
            round_idx=round_idx,
            attack_name=attack_name,
            sampled_clients=sampled_clients,
            benign_clients=benign_clients,
            selected_attackers=selected_attackers,
            clean_loss=clean_loss,
            clean_acc=clean_acc,
            round_seconds=round_seconds,
            client_metrics=list(client_metrics.values()),
            benign_update_norms=benign_norms,
            malicious_update_norms=malicious_norms,
            malicious_cosines_to_benign=malicious_cosines,
        )

    def run_many_rounds(
        self,
        rounds: int,
        attack: Optional[SandboxAttack] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        eval_every: int = 1,
    ) -> List[RoundSummary]:
        """Run multiple rounds and collect summaries."""
        summaries = []
        iterator = range(1, rounds + 1)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, total=rounds, desc=progress_desc or "FL rounds", unit="round")
        for round_idx in iterator:
            should_evaluate = eval_every <= 1 or round_idx % eval_every == 0 or round_idx == rounds
            summary = self.run_round(round_idx, attack=attack, evaluate=should_evaluate)
            summaries.append(summary)
            if show_progress and tqdm is not None:
                postfix = {"sec": f"{summary.round_seconds:.2f}"}
                if should_evaluate:
                    postfix["acc"] = f"{summary.clean_acc:.4f}"
                iterator.set_postfix(**postfix)
        return summaries

    def _train_client(self, old_weights, client_id: int) -> tuple[List[np.ndarray], float, float]:
        self._load_numpy_weights(self.client_model, old_weights)
        train_loss, train_acc = self._local_train(self.client_model, self.client_loaders[client_id])
        return self._capture_weights(self.client_model), train_loss, train_acc

    def _train_clients_parallel(self, old_weights, client_ids: List[int]) -> List[tuple[int, List[np.ndarray], float, float]]:
        max_workers = min(self.config.parallel_clients, len(client_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._train_client_isolated, old_weights, client_id)
                for client_id in client_ids
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda item: item[0])
        return results

    def _train_client_isolated(self, old_weights, client_id: int) -> tuple[int, List[np.ndarray], float, float]:
        model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._load_numpy_weights(model, old_weights)
        train_loss, train_acc = self._local_train(
            model,
            self.client_loaders[client_id],
            optimizer=optimizer,
            scaler=scaler,
        )
        return client_id, self._capture_weights(model), train_loss, train_acc

    def _local_train(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> tuple[float, float]:
        model.train()
        runtime_optimizer = optimizer or self.client_optimizer
        runtime_scaler = scaler or self.scaler
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for _ in range(self.config.local_epochs):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=self.device.type == "cuda")
                labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
                if self.device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                runtime_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=True,
                    ):
                        logits = model(images)
                        loss = self.criterion(logits, labels)
                    runtime_scaler.scale(loss).backward()
                    runtime_scaler.step(runtime_optimizer)
                    runtime_scaler.update()
                else:
                    logits = model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    runtime_optimizer.step()
                batch_size = labels.size(0)
                total_loss += loss.detach().item() * batch_size
                total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            return 0.0, 0.0
        return total_loss / total_examples, total_correct / total_examples

    def _average_weights(self, weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        return [
            np.mean([weights[layer] for weights in weights_list], axis=0)
            for layer in range(len(weights_list[0]))
        ]

    def _build_model(self) -> torch.nn.Module:
        if self.config.dataset == "cifar10":
            return ResNet18()
        return MNISTClassifier()

    @staticmethod
    def _capture_weights(model: torch.nn.Module) -> List[np.ndarray]:
        weights: List[np.ndarray] = []
        for value in model.state_dict().values():
            weights.append(value.detach().cpu().numpy().copy())
        return weights

    def _load_numpy_weights(self, model: torch.nn.Module, weights: List[np.ndarray]) -> None:
        with torch.no_grad():
            for target, source in zip(model.state_dict().values(), weights):
                source_tensor = torch.as_tensor(source, device=target.device, dtype=target.dtype)
                target.copy_(source_tensor.reshape_as(target))

    def _split_data(self) -> List[set]:
        num_items = len(self.train_dataset) // self.config.num_clients
        all_indices = list(range(len(self.train_dataset)))
        client_data_idxs = []
        rng = random.Random(self.config.seed)
        for _ in range(self.config.num_clients):
            chosen = set(rng.sample(all_indices, min(num_items, len(all_indices))))
            client_data_idxs.append(chosen)
            all_indices = list(set(all_indices) - chosen)
        return client_data_idxs

    def _sample_clients(self, round_idx: int) -> List[int]:
        num_sampled = max(1, int(self.config.num_clients * self.config.subsample_rate))
        rng = random.Random(self.config.seed + round_idx * 997)
        return sorted(rng.sample(range(self.config.num_clients), num_sampled))

    def _make_loader_kwargs(self, shuffle: bool) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "shuffle": shuffle,
            "pin_memory": self.device.type == "cuda",
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return kwargs

    @staticmethod
    def _resolve_num_workers(configured_workers: Optional[int]) -> int:
        if configured_workers is not None:
            return max(0, configured_workers)
        cpu_count = os.cpu_count() or 1
        return min(8, max(2, cpu_count // 2))

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def summaries_to_dict(summaries: List[RoundSummary]) -> Dict[str, List[float]]:
    """Flatten summaries into round-wise scalar series."""
    return {
        "clean_loss": [summary.clean_loss for summary in summaries],
        "clean_acc": [summary.clean_acc for summary in summaries],
        "round_seconds": [summary.round_seconds for summary in summaries],
        "mean_benign_norm": [
            float(np.mean(summary.benign_update_norms)) if summary.benign_update_norms else 0.0
            for summary in summaries
        ],
        "mean_malicious_norm": [
            float(np.mean(summary.malicious_update_norms)) if summary.malicious_update_norms else 0.0
            for summary in summaries
        ],
        "mean_malicious_cosine": [
            float(np.mean(summary.malicious_cosines_to_benign))
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
