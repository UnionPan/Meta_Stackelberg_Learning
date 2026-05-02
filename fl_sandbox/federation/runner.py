"""Minimal federated-learning runner for attacker validation."""

from __future__ import annotations

import copy
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl_sandbox.data import DatasetSplit, get_datasets, poison_dataset
from fl_sandbox.evaluation import test_model
from fl_sandbox.federation.client import ClientTrainer
from fl_sandbox.models import build_model
from fl_sandbox.utils import resolve_device, set_parameters

from fl_sandbox.attacks.base import SandboxAttack
from fl_sandbox.aggregators.rules import AggregationDefender, PaperActionDefender
from fl_sandbox.core.metrics import update_norm
from fl_sandbox.core.runtime import (
    RoundRuntimeState,
    RoundSummary,
    RoundTimer,
    build_round_context,
    build_round_summary,
    summarize_round_updates,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
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
    max_client_samples_per_client: Optional[int] = None
    max_eval_samples: Optional[int] = None
    num_workers: Optional[int] = None
    prefetch_factor: int = 4
    parallel_clients: int = 1
    base_class: int = 1
    target_class: int = 7
    pattern_type: str = "square"
    ipm_scaling: float = 2.0
    lmp_scale: float = 2.0
    bfl_poison_frac: float = 1.0
    dba_poison_frac: float = 0.5
    dba_num_sub_triggers: int = 4
    attacker_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    defense_type: str = "fedavg"
    krum_attackers: int = 1
    multi_krum_selected: Optional[int] = None
    clipped_median_norm: float = 2.0
    trimmed_mean_ratio: float = 0.2
    geometric_median_iters: int = 10
    fltrust_root_size: int = 100
    split_mode: str = "iid"
    noniid_q: float = 0.5
    rl_distribution_steps: int = 10
    rl_attack_start_round: int = 10
    rl_policy_train_end_round: int = 30
    rl_inversion_steps: int = 50
    rl_reconstruction_batch_size: int = 8
    rl_policy_train_episodes_per_round: int = 1
    rl_simulator_horizon: int = 8
    init_mode: str = "seed"
    init_checkpoint_path: str = ""


class MinimalFLRunner:
    """Compact FL runner for attacker-side validation."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.device = resolve_device(self.config.device)
        self.num_workers = self._resolve_num_workers(self.config.num_workers)
        self.loader_kwargs = self._make_loader_kwargs(shuffle=True)
        self.eval_loader_kwargs = self._make_loader_kwargs(shuffle=False)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.defender = AggregationDefender(
            defense_type=self.config.defense_type,
            krum_attackers=self.config.krum_attackers,
            multi_krum_selected=self.config.multi_krum_selected,
            clipped_median_norm=self.config.clipped_median_norm,
            trimmed_mean_ratio=self.config.trimmed_mean_ratio,
            geometric_median_iters=self.config.geometric_median_iters,
        )
        self._set_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.train_dataset, self.test_dataset = get_datasets(self.config.dataset, data_dir=self.config.data_dir)
        self.eval_indices = self._eval_indices()
        self.test_loader = DataLoader(
            DatasetSplit(self.test_dataset, self.eval_indices),
            batch_size=self.config.eval_batch_size,
            **self.eval_loader_kwargs,
        )
        self.client_groups = self._assign_client_groups()
        self.client_data_idxs = self._limit_client_data_idxs(self._split_data())
        self.attacker_ids = self._assign_attacker_ids()
        self.root_loader = self._prepare_root_loader()
        self.client_loaders = [
            DataLoader(
                DatasetSplit(self.train_dataset, list(self.client_data_idxs[client_id])),
                batch_size=self.config.batch_size,
                **self.loader_kwargs,
            )
            for client_id in range(self.config.num_clients)
        ]
        self.model, self.client_model = self._initialize_model_pair()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=self.config.lr)
        self.current_weights = self._capture_weights(self.model)
        self.poisoned_train_loaders = self._prepare_poisoned_train_loaders()
        self.poisoned_eval_loader = self._prepare_poisoned_eval_loader()

    def reset_model(self) -> None:
        self._set_seed(self.config.seed)
        self.model, self.client_model = self._initialize_model_pair()
        self.client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=self.config.lr)
        self.current_weights = self._capture_weights(self.model)
        self.poisoned_train_loaders = self._prepare_poisoned_train_loaders()
        self.poisoned_eval_loader = self._prepare_poisoned_eval_loader()

    def run_round(
        self,
        round_idx: int,
        attack: Optional[SandboxAttack] = None,
        evaluate: bool = True,
        attacker_action: Optional[np.ndarray] = None,
        defense_decision: object | None = None,
    ) -> RoundSummary:
        round_timer = RoundTimer.start()
        old_weights = [weights.copy() for weights in self.current_weights]
        sampled_clients = self._sample_clients(round_idx)
        round_state = RoundRuntimeState.from_selection(
            round_idx=round_idx,
            sampled_clients=sampled_clients,
            attacker_ids=self.attacker_ids,
            num_clients=self.config.num_clients,
        )
        all_attacker_loader = self._build_attacker_loader(self.attacker_ids)
        selected_attacker_loader = self._build_attacker_loader(round_state.selected_attackers)
        selected_attacker_train_loaders = {
            attacker_id: self.client_loaders[attacker_id]
            for attacker_id in round_state.selected_attackers
        }
        trusted_weights = self._trusted_reference_update(old_weights)

        benign_weights = []
        if round_state.benign_clients:
            if self.config.parallel_clients > 1:
                for cid, weights, train_loss, train_acc in self._train_clients_parallel(
                    old_weights, round_state.benign_clients
                ):
                    benign_weights.append(weights)
                    round_state.record_client_training(
                        cid, train_loss=train_loss, train_acc=train_acc, update_norm=update_norm(old_weights, weights)
                    )
            else:
                for cid in round_state.benign_clients:
                    weights, train_loss, train_acc = self._train_client(old_weights, cid)
                    benign_weights.append(weights)
                    round_state.record_client_training(
                        cid, train_loss=train_loss, train_acc=train_acc, update_norm=update_norm(old_weights, weights)
                    )
        all_weights = list(benign_weights)

        malicious_weights: List[List[np.ndarray]] = []
        attack_name = attack.name if attack is not None else "clean"
        if attack is not None:
            ctx = build_round_context(
                round_idx=round_idx,
                old_weights=old_weights,
                benign_weights=benign_weights,
                selected_attacker_ids=round_state.selected_attackers,
                model=self.model,
                device=self.device,
                fl_config=self.config,
                defense_type=self._round_defender(defense_decision).defense_type,
                lr=self.config.lr,
                local_epochs=self.config.local_epochs,
                attacker_train_iter=selected_attacker_loader,
                all_attacker_train_iter=all_attacker_loader,
                selected_attacker_train_loaders=selected_attacker_train_loaders,
                global_poisoned_train_loader=self.poisoned_train_loaders.get("global"),
                sub_trigger_train_loaders=self.poisoned_train_loaders.get("sub_triggers"),
                poisoned_train_iters=self.poisoned_train_loaders,
                attacker_action=np.asarray(
                    attacker_action if attacker_action is not None else self.config.attacker_action,
                    dtype=float,
                ),
                trusted_reference_weights=trusted_weights,
            )
            attack.observe_round(ctx)
            if round_state.selected_attackers:
                malicious_weights = attack.execute(ctx, attacker_action=attacker_action)
                all_weights.extend(malicious_weights)

        if all_weights:
            round_defender = self._round_defender(defense_decision)
            self.current_weights = round_defender.aggregate(
                old_weights, all_weights, trusted_weights=trusted_weights
            )
            set_parameters(self.model, self.current_weights)

        if evaluate:
            clean_loss, clean_acc = test_model(self.model, self.test_loader, device=self.device)
            if self.poisoned_eval_loader is not None:
                _, backdoor_acc = test_model(self.model, self.poisoned_eval_loader, device=self.device)
            else:
                backdoor_acc = float("nan")
        else:
            clean_loss, clean_acc, backdoor_acc = float("nan"), float("nan"), float("nan")

        update_stats = summarize_round_updates(old_weights, benign_weights, malicious_weights)
        return build_round_summary(
            state=round_state,
            attack_name=attack_name,
            defense_name=self._round_defender(defense_decision).defense_type,
            clean_loss=clean_loss,
            clean_acc=clean_acc,
            backdoor_acc=backdoor_acc,
            round_seconds=round_timer.elapsed_seconds(),
            update_stats=update_stats,
        )

    def run_many_rounds(
        self,
        rounds: int,
        attack: Optional[SandboxAttack] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        eval_every: int = 1,
        attacker_action: Optional[np.ndarray] = None,
        defense_decision: object | None = None,
        per_round_callback: Optional[Callable[[RoundSummary], None]] = None,
    ) -> List[RoundSummary]:
        summaries = []
        progress_label = progress_desc or "FL rounds"
        use_tqdm = show_progress and tqdm is not None and self._supports_live_progress()
        iterator = range(1, rounds + 1)
        if use_tqdm:
            iterator = tqdm(iterator, total=rounds, desc=progress_label, unit="round", file=sys.stdout, dynamic_ncols=True)
        elif show_progress:
            print(f"{progress_label}: starting {rounds} rounds", flush=True)
        for round_idx in iterator:
            should_evaluate = eval_every <= 1 or round_idx % eval_every == 0 or round_idx == rounds
            summary = self.run_round(
                round_idx,
                attack=attack,
                evaluate=should_evaluate,
                attacker_action=attacker_action,
                defense_decision=defense_decision,
            )
            summaries.append(summary)
            if per_round_callback is not None:
                per_round_callback(summary)
            if use_tqdm:
                postfix = {"sec": f"{summary.round_seconds:.2f}"}
                if should_evaluate:
                    postfix["acc"] = f"{summary.clean_acc:.4f}"
                iterator.set_postfix(**postfix)
            elif show_progress:
                self._print_progress_line(
                    progress_label=progress_label,
                    round_idx=round_idx,
                    rounds=rounds,
                    summary=summary,
                    evaluated=should_evaluate,
                )
        return summaries

    def _round_defender(self, defense_decision: object | None):
        if defense_decision is None:
            return self.defender
        alpha = float(getattr(defense_decision, "norm_bound_alpha"))
        beta = float(getattr(defense_decision, "trimmed_mean_beta"))
        return PaperActionDefender(norm_bound_alpha=alpha, trimmed_mean_beta=beta)

    @staticmethod
    def _supports_live_progress() -> bool:
        output_stream = getattr(sys, "stdout", None)
        if output_stream is None or not hasattr(output_stream, "isatty"):
            return False
        try:
            return bool(output_stream.isatty())
        except Exception:
            return False

    @staticmethod
    def _print_progress_line(*, progress_label, round_idx, rounds, summary, evaluated) -> None:
        metrics = [f"sec={summary.round_seconds:.2f}"]
        if evaluated:
            metrics.append(f"acc={summary.clean_acc:.4f}")
            if not np.isnan(summary.backdoor_acc):
                metrics.append(f"asr={summary.backdoor_acc:.4f}")
        print(f"{progress_label}: round {round_idx}/{rounds} ({' '.join(metrics)})", flush=True)

    def _train_client(self, old_weights, client_id: int) -> tuple[List[np.ndarray], float, float]:
        self._load_numpy_weights(self.client_model, old_weights)
        train_loss, train_acc = self._local_train(self.client_model, self.client_loaders[client_id])
        return self._capture_weights(self.client_model), train_loss, train_acc

    def _train_clients_parallel(self, old_weights, client_ids: List[int]):
        max_workers = min(self.config.parallel_clients, len(client_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._train_client_isolated, old_weights, cid) for cid in client_ids]
            results = [f.result() for f in futures]
        results.sort(key=lambda item: item[0])
        return results

    def _train_client_isolated(self, old_weights, client_id: int):
        model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._load_numpy_weights(model, old_weights)
        train_loss, train_acc = self._local_train(model, self.client_loaders[client_id], optimizer=optimizer, scaler=scaler)
        return client_id, self._capture_weights(model), train_loss, train_acc

    def _local_train(self, model, loader, optimizer=None, scaler=None) -> tuple[float, float]:
        trainer = ClientTrainer(
            criterion=self.criterion,
            device=self.device,
            lr=self.config.lr,
            local_epochs=self.config.local_epochs,
            use_amp=self.use_amp,
        )
        return trainer.train(model, loader, optimizer=optimizer or self.client_optimizer, scaler=scaler or self.scaler)

    def _prepare_root_loader(self) -> Optional[DataLoader]:
        if self.config.fltrust_root_size <= 0:
            return None
        root_size = min(self.config.fltrust_root_size, len(self.train_dataset))
        if root_size <= 0:
            return None
        rng = random.Random(self.config.seed + 2024)
        root_indices = rng.sample(range(len(self.train_dataset)), root_size)
        return DataLoader(
            DatasetSplit(self.train_dataset, root_indices),
            batch_size=min(self.config.batch_size, root_size),
            **self.loader_kwargs,
        )

    def _trusted_reference_update(self, old_weights) -> Optional[List[np.ndarray]]:
        if self.defender.defense_type != "fltrust" or self.root_loader is None:
            return None
        model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._load_numpy_weights(model, old_weights)
        self._local_train(model, self.root_loader, optimizer=optimizer, scaler=scaler)
        return self._capture_weights(model)

    def _prepare_poisoned_train_loaders(self) -> Dict[str, object]:
        if self.config.num_attackers <= 0:
            return {}
        loaders: Dict[str, object] = {"global_by_attacker": {}, "sub_triggers_by_attacker": {}}
        attacker_union_indices = sorted(
            idx for attacker_id in self.attacker_ids for idx in self.client_data_idxs[attacker_id]
        )
        if attacker_union_indices:
            global_poisoned_dataset = copy.deepcopy(self.train_dataset)
            poison_dataset(
                global_poisoned_dataset, self.config.dataset, self.config.base_class, self.config.target_class,
                poison_frac=self.config.bfl_poison_frac, pattern_type=self.config.pattern_type,
                data_idxs=attacker_union_indices, poison_all=self.config.bfl_poison_frac >= 1.0,
            )
            loaders["global"] = DataLoader(
                DatasetSplit(global_poisoned_dataset, attacker_union_indices),
                batch_size=self.config.batch_size, **self.loader_kwargs,
            )
            sub_trigger_loaders = []
            for sub_idx in range(max(1, self.config.dba_num_sub_triggers)):
                sub_poisoned_dataset = copy.deepcopy(self.train_dataset)
                poison_dataset(
                    sub_poisoned_dataset, self.config.dataset, self.config.base_class, self.config.target_class,
                    poison_frac=self.config.dba_poison_frac, pattern_type=self.config.pattern_type,
                    data_idxs=attacker_union_indices, agent_idx=sub_idx,
                )
                sub_trigger_loaders.append(DataLoader(
                    DatasetSplit(sub_poisoned_dataset, attacker_union_indices),
                    batch_size=self.config.batch_size, **self.loader_kwargs,
                ))
            loaders["sub_triggers"] = sub_trigger_loaders

        for attacker_id in self.attacker_ids:
            local_indices = sorted(self.client_data_idxs[attacker_id])
            if not local_indices:
                continue
            local_global_dataset = copy.deepcopy(self.train_dataset)
            poison_dataset(
                local_global_dataset, self.config.dataset, self.config.base_class, self.config.target_class,
                poison_frac=self.config.bfl_poison_frac, pattern_type=self.config.pattern_type,
                data_idxs=local_indices, poison_all=self.config.bfl_poison_frac >= 1.0,
            )
            loaders["global_by_attacker"][attacker_id] = DataLoader(
                DatasetSplit(local_global_dataset, local_indices),
                batch_size=self.config.batch_size, **self.loader_kwargs,
            )
            local_sub_trigger_loaders = []
            for sub_idx in range(max(1, self.config.dba_num_sub_triggers)):
                local_sub_dataset = copy.deepcopy(self.train_dataset)
                poison_dataset(
                    local_sub_dataset, self.config.dataset, self.config.base_class, self.config.target_class,
                    poison_frac=self.config.dba_poison_frac, pattern_type=self.config.pattern_type,
                    data_idxs=local_indices, agent_idx=sub_idx,
                )
                local_sub_trigger_loaders.append(DataLoader(
                    DatasetSplit(local_sub_dataset, local_indices),
                    batch_size=self.config.batch_size, **self.loader_kwargs,
                ))
            loaders["sub_triggers_by_attacker"][attacker_id] = local_sub_trigger_loaders
        return loaders

    def _prepare_poisoned_eval_loader(self) -> Optional[DataLoader]:
        targets = self.test_dataset.targets
        if isinstance(targets, torch.Tensor):
            base_idxs = targets.eq(self.config.base_class).nonzero(as_tuple=True)[0].tolist()
        else:
            base_idxs = [idx for idx, t in enumerate(targets) if int(t) == self.config.base_class]
        if not base_idxs:
            return None
        if self.config.max_eval_samples is not None and self.config.max_eval_samples > 0:
            base_idxs = base_idxs[: min(len(base_idxs), self.config.max_eval_samples)]
        poisoned_eval_dataset = copy.deepcopy(self.test_dataset)
        poison_dataset(
            poisoned_eval_dataset, self.config.dataset, self.config.base_class, self.config.target_class,
            poison_frac=1.0, pattern_type=self.config.pattern_type, poison_all=True,
        )
        return DataLoader(
            DatasetSplit(poisoned_eval_dataset, base_idxs),
            batch_size=self.config.eval_batch_size, **self.eval_loader_kwargs,
        )

    def _build_attacker_loader(self, selected_attackers: List[int]) -> Optional[DataLoader]:
        if not selected_attackers:
            return None
        attacker_indices = sorted(idx for cid in selected_attackers for idx in self.client_data_idxs[cid])
        if not attacker_indices:
            return None
        return DataLoader(
            DatasetSplit(self.train_dataset, attacker_indices),
            batch_size=self.config.batch_size, **self.loader_kwargs,
        )

    def _build_model(self) -> torch.nn.Module:
        return build_model(self.config.dataset)

    def _initialize_model_pair(self):
        model = self._build_model().to(self.device)
        client_model = self._build_model().to(self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            client_model = client_model.to(memory_format=torch.channels_last)
        if self.config.init_mode == "checkpoint":
            self._load_checkpoint_into_model(model, self.config.init_checkpoint_path)
            self._load_checkpoint_into_model(client_model, self.config.init_checkpoint_path)
        elif self.config.init_mode != "seed":
            raise ValueError(f"Unsupported init_mode: {self.config.init_mode}")
        return model, client_model

    def _load_checkpoint_into_model(self, model, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = self._extract_state_dict_payload(model, checkpoint)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
            return
        if isinstance(checkpoint, (list, tuple)):
            weights = [np.asarray(layer) for layer in checkpoint]
            self._load_numpy_weights(model, weights)
            return
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_file}.")

    def _extract_state_dict_payload(self, model, checkpoint):
        if isinstance(checkpoint, torch.nn.Module):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            state_dict = None
            for key in ("state_dict", "model_state_dict"):
                payload = checkpoint.get(key)
                if isinstance(payload, dict):
                    state_dict = payload
                    break
            if state_dict is None and self._looks_like_state_dict(model, checkpoint):
                state_dict = checkpoint
        else:
            return None
        return self._normalize_state_dict_keys(model, state_dict)

    @staticmethod
    def _looks_like_state_dict(model, payload) -> bool:
        expected_keys = set(model.state_dict().keys())
        payload_keys = {str(key) for key in payload.keys()}
        if payload_keys == expected_keys:
            return True
        stripped_keys = {key[len("module."):] if key.startswith("module.") else key for key in payload_keys}
        return stripped_keys == expected_keys

    @staticmethod
    def _normalize_state_dict_keys(model, state_dict):
        expected_keys = list(model.state_dict().keys())
        normalized = {
            (str(key)[len("module."):] if str(key).startswith("module.") else str(key)): value
            for key, value in state_dict.items()
        }
        if set(normalized.keys()) != set(expected_keys):
            raise ValueError("Checkpoint state_dict keys do not match model architecture.")
        model_state = model.state_dict()
        return {key: torch.as_tensor(normalized[key], dtype=model_state[key].dtype) for key in expected_keys}

    @staticmethod
    def _capture_weights(model) -> List[np.ndarray]:
        return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]

    def _load_numpy_weights(self, model, weights: List[np.ndarray]) -> None:
        with torch.no_grad():
            for target, source in zip(model.state_dict().values(), weights):
                source_tensor = torch.as_tensor(source, device=target.device, dtype=target.dtype)
                target.copy_(source_tensor.reshape_as(target))

    def _split_data(self) -> List[set]:
        mode = self.config.split_mode.lower()
        if mode == "iid":
            return self._split_data_iid()
        if mode == "noniid":
            return self._split_data_noniid()
        if mode == "paper_q":
            return self._split_data_paper_q()
        raise ValueError(f"Unsupported split_mode: {self.config.split_mode}")

    def _split_data_iid(self) -> List[set]:
        num_items = len(self.train_dataset) // self.config.num_clients
        all_indices = list(range(len(self.train_dataset)))
        client_data_idxs = []
        rng = random.Random(self.config.seed)
        for _ in range(self.config.num_clients):
            chosen = set(rng.sample(all_indices, min(num_items, len(all_indices))))
            client_data_idxs.append(chosen)
            all_indices = list(set(all_indices) - chosen)
        return client_data_idxs

    def _limit_client_data_idxs(self, client_data_idxs: List[set]) -> List[set]:
        max_samples = self.config.max_client_samples_per_client
        if max_samples is None or max_samples <= 0:
            return client_data_idxs
        rng = random.Random(self.config.seed + 3030)
        limited = []
        for indices in client_data_idxs:
            ordered = sorted(indices)
            if len(ordered) > max_samples:
                ordered = rng.sample(ordered, max_samples)
            limited.append(set(ordered))
        return limited

    def _eval_indices(self) -> List[int]:
        indices = list(range(len(self.test_dataset)))
        max_samples = self.config.max_eval_samples
        if max_samples is None or max_samples <= 0 or max_samples >= len(indices):
            return indices
        rng = random.Random(self.config.seed + 4040)
        return sorted(rng.sample(indices, max_samples))

    def _split_data_noniid(self) -> List[set]:
        targets = self._dataset_targets()
        classes = sorted({int(label) for label in targets})
        num_groups = len(classes)
        if self.config.num_clients < num_groups:
            raise ValueError("split_mode='noniid' requires num_clients >= number of classes")
        class_to_group = {label: idx for idx, label in enumerate(classes)}
        group_to_clients = {group_id: [] for group_id in range(num_groups)}
        for client_id, group_id in enumerate(self.client_groups):
            group_to_clients[group_id].append(client_id)
        rng = random.Random(self.config.seed)
        client_data_idxs = [set() for _ in range(self.config.num_clients)]
        grouped_indices = {group_id: [] for group_id in range(num_groups)}
        q = max(1.0 / num_groups, min(1.0, float(self.config.noniid_q)))
        for idx, label in enumerate(targets):
            preferred_group = class_to_group[int(label)]
            if rng.random() < q:
                assigned_group = preferred_group
            else:
                other_groups = [gid for gid in range(num_groups) if gid != preferred_group]
                assigned_group = rng.choice(other_groups) if other_groups else preferred_group
            grouped_indices[assigned_group].append(idx)
        for group_id, indices in grouped_indices.items():
            clients = group_to_clients[group_id]
            if not clients:
                continue
            rng.shuffle(indices)
            for offset, sample_idx in enumerate(indices):
                client_data_idxs[clients[offset % len(clients)]].add(sample_idx)
        return client_data_idxs

    def _split_data_paper_q(self) -> List[set]:
        targets = self._dataset_targets()
        classes = sorted({int(label) for label in targets})
        num_groups = len(classes)
        if self.config.num_clients < num_groups:
            raise ValueError("split_mode='paper_q' requires num_clients >= number of classes")
        class_to_group = {label: idx for idx, label in enumerate(classes)}
        group_to_clients = {group_id: [] for group_id in range(num_groups)}
        for client_id, group_id in enumerate(self.client_groups):
            group_to_clients[group_id].append(client_id)
        rng = random.Random(self.config.seed)
        client_data_idxs = [set() for _ in range(self.config.num_clients)]
        grouped_indices = {group_id: [] for group_id in range(num_groups)}
        base_iid = 1.0 / max(1, num_groups)
        q = max(base_iid, min(1.0, float(self.config.noniid_q)))
        keep_label_prob = (q - base_iid) * num_groups / max(1, num_groups - 1)
        for idx, label in enumerate(targets):
            label_group = class_to_group[int(label)]
            assigned_group = label_group if rng.random() < keep_label_prob else rng.randrange(num_groups)
            grouped_indices[assigned_group].append(idx)
        for group_id, indices in grouped_indices.items():
            clients = group_to_clients[group_id]
            if not clients:
                continue
            for offset, sample_idx in enumerate(indices):
                client_data_idxs[clients[offset % len(clients)]].add(sample_idx)
        return client_data_idxs

    def _assign_client_groups(self) -> List[int]:
        targets = self._dataset_targets()
        classes = sorted({int(label) for label in targets})
        num_groups = max(1, len(classes))
        return [client_id % num_groups for client_id in range(self.config.num_clients)]

    def _assign_attacker_ids(self) -> List[int]:
        if self.config.num_attackers <= 0:
            return []
        group_to_clients: Dict[int, List[int]] = {}
        for client_id, group_id in enumerate(self.client_groups):
            group_to_clients.setdefault(group_id, []).append(client_id)
        for clients in group_to_clients.values():
            clients.sort()
        attacker_ids: List[int] = []
        group_ids = sorted(group_to_clients)
        cursor = 0
        while len(attacker_ids) < self.config.num_attackers:
            group_id = group_ids[cursor % len(group_ids)]
            clients = group_to_clients[group_id]
            pick_idx = len([cid for cid in attacker_ids if self.client_groups[cid] == group_id])
            if pick_idx < len(clients):
                attacker_ids.append(clients[pick_idx])
            cursor += 1
            if cursor > self.config.num_clients * 4:
                break
        return sorted(attacker_ids[: self.config.num_attackers])

    def _dataset_targets(self) -> List[int]:
        targets = self.train_dataset.targets
        if isinstance(targets, torch.Tensor):
            return [int(value) for value in targets.cpu().tolist()]
        return [int(value) for value in list(targets)]

    def _sample_clients(self, round_idx: int) -> List[int]:
        num_sampled = min(self.config.num_clients, max(1, int(self.config.num_clients * self.config.subsample_rate)))
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
