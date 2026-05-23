"""Paper-style RL backdoor attacker.

Owns its TD3 policy and the grey-box ``SimulatedBackdoorFLEnv`` it trains in.
Lifecycle is the same as every other ``SandboxAttack`` — the runner calls
``observe_round`` then ``execute`` then ``after_round`` per round. Policy
training happens inside ``observe_round`` between ``attack_start_round`` and
``policy_train_end_round``; deployment is just ``trainer.act(obs)`` followed
by the Phase 1 sybil-broadcast crafting. The runner is never wrapped in a
Gym env — control flow stays right-side up.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from fl_sandbox.aggregators.rules import AggregationDefender
from fl_sandbox.attacks.base import (
    SandboxAttack,
    Weights,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.attacks.rl_attacker.trainer import Trainer, build_trainer
from fl_sandbox.attacks.rl_backdoor.action import BackdoorAction, decode_backdoor_action
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder
from fl_sandbox.attacks.rl_backdoor.simulator import SimulatedBackdoorFLEnv
from fl_sandbox.core.metrics import update_norm


def _fl_value(fl_config, section: str, name: str, fallback):
    section_obj = getattr(fl_config, section, None)
    if section_obj is not None and hasattr(section_obj, name):
        return getattr(section_obj, name)
    return getattr(fl_config, name, fallback)


@dataclass
class RLBackdoorAttack(SandboxAttack):
    """Backdoor attacker controlled by a TD3 policy trained in a grey-box simulator.

    See ``fl_sandbox/docs/superpowers/plans/2026-05-22-rl-backdoor-attacker-paper-faithful-plan.md``
    §2-§5 for the design: model-free agent, model-based grey-box training,
    paper reward (negative weighted clean/backdoor shadow loss with ``γ=1``),
    observable-only state, ``truncated`` (not ``terminated``) at horizon.

    Falls back to the static-default-action behaviour from Phase 1 whenever the
    policy isn't trained yet — same crafting (poison-rate grid + sybil
    average + boost broadcast), so the live path is well-defined even with
    ``attack_start_round`` not yet reached.
    """

    default_action: tuple[float, float, float, float] = (1.0, 0.0, -1.0, 0.0)
    stealth_norm_cap: bool = False
    config: Optional[BackdoorRLConfig] = None
    attack_start_round: int = 10
    policy_train_end_round: int = 30
    policy_train_steps_per_round: int = 50
    name: str = "RLBackdoor"
    attack_type: str = "rl_backdoor"

    # internal state (not dataclass init args)
    _trainer: Optional[Trainer] = field(default=None, init=False, repr=False)
    _simulator: Optional[SimulatedBackdoorFLEnv] = field(default=None, init=False, repr=False)
    _observation_builder: Optional[BackdoorObservationBuilder] = field(default=None, init=False, repr=False)
    _model_template: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _previous_global_weights: Optional[Weights] = field(default=None, init=False, repr=False)
    _latest_global_weights: Optional[Weights] = field(default=None, init=False, repr=False)
    _last_action: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_local_bd_success: float = field(default=0.0, init=False, repr=False)
    _last_obs: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _diagnostics: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _checkpoint_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = self.config or BackdoorRLConfig()
        self._observation_builder = BackdoorObservationBuilder(self.config)
        self._last_action = np.zeros(self.config.action_dim, dtype=np.float32)

    # ----------------------------------------------------------- lifecycle

    def observe_round(self, ctx) -> None:
        # Per-round training counters must not bleed into the next round's
        # diagnostics — clear them up front so ``after_round`` reports zero
        # when training did not actually fire this round. Keys mirror the
        # ``rl_*`` namespace that ``experiment_service.rl_training_tensorboard_scalars``
        # already maps to TensorBoard so the same dashboards work for both
        # ``rl`` and ``rl_backdoor`` without special-casing.
        for key in list(self._diagnostics):
            if key.startswith("rl_trainer_") or key.startswith("rl_simulated_"):
                self._diagnostics.pop(key, None)
        if torch is None or ctx.model is None or ctx.device is None:
            return
        if self._model_template is None:
            self._model_template = copy.deepcopy(ctx.model).to("cpu")
        self._device = ctx.device

        self._previous_global_weights = (
            self._latest_global_weights or [layer.copy() for layer in ctx.old_weights]
        )
        self._latest_global_weights = [layer.copy() for layer in ctx.old_weights]
        self._last_local_bd_success = self._evaluate_local_bd_success(ctx, ctx.old_weights)

        if self.config.freeze_policy:
            self._maybe_load_checkpoint(ctx.round_idx)
            return
        if not (self.attack_start_round <= ctx.round_idx <= self.policy_train_end_round):
            return
        if not self._ensure_simulator(ctx):
            return
        self._maybe_load_checkpoint(ctx.round_idx)
        self._train_policy_one_round()

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        num_attackers = self.selected_attacker_count(ctx)
        if num_attackers == 0:
            return []

        # Warmup contract: before ``attack_start_round`` the attacker observes
        # the FL state but does NOT poison — same as ``RLAttack`` — so the FL
        # baseline can converge before the policy deploys. Without this guard
        # the static default action would inject a malicious update from
        # round 1, contaminating the warmup phase. ``round_idx`` may be absent
        # on synthetic unit-test contexts; those skip the warmup gate.
        round_idx = getattr(ctx, "round_idx", None)
        if (
            round_idx is not None
            and int(round_idx) < int(self.attack_start_round)
            and attacker_action is None
        ):
            return self._fallback_benign_weights(ctx, num_attackers)

        action = self._resolve_action(ctx, attacker_action)
        decoded = decode_backdoor_action(action)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)[: self.config.action_dim]

        return self._craft_sybil_broadcast(ctx, decoded, num_attackers)

    def _fallback_benign_weights(self, ctx, num_attackers: int) -> List[Weights]:
        """Train each sampled attacker on its clean client loader and submit honestly."""
        fallback: List[Weights] = []
        attacker_loaders = getattr(ctx, "selected_attacker_train_loaders", None) or {}
        for attacker_id in ctx.selected_attacker_ids:
            loader = attacker_loaders.get(attacker_id)
            if loader is None:
                fallback.append(self.clone_old_weights(ctx))
                continue
            fallback.append(train_on_loader(ctx, loader))
        if len(fallback) < num_attackers:
            fallback.extend(self.clone_old_weights(ctx) for _ in range(num_attackers - len(fallback)))
        return fallback

    def after_round(self, **kwargs) -> Dict[str, float]:
        live_poi_acc = float(self._last_local_bd_success)
        sim_poi_acc = float(self._diagnostics.get("rl_simulated_poi_acc", 0.0))
        backdoor_acc = float(kwargs.get("backdoor_acc", float("nan")))
        diagnostics = dict(self._diagnostics)
        diagnostics["rl_backdoor_live_poi_acc"] = live_poi_acc
        diagnostics["rl_backdoor_global_asr"] = backdoor_acc
        if np.isfinite(backdoor_acc):
            # Transfer-gap metric: |shadow trigger-set accuracy - real global ASR|.
            # Reused by Phase 5 reporting. Picked up by the standard
            # ``rl_sim2real_gap`` TensorBoard scalar without special-casing.
            diagnostics["rl_sim2real_gap"] = float(abs(sim_poi_acc - backdoor_acc))
        return diagnostics

    # ----------------------------------------------------- action selection

    def _resolve_action(self, ctx, attacker_action) -> np.ndarray:
        if attacker_action is not None:
            return np.asarray(attacker_action, dtype=np.float32).reshape(-1)
        if self._trainer is not None and getattr(self._trainer, "policy", None) is not None:
            obs = self._build_live_observation(ctx)
            self._last_obs = obs
            deterministic = bool(self.config.freeze_policy) or ctx.round_idx > self.policy_train_end_round
            action = np.asarray(self._trainer.act(obs, deterministic=deterministic), dtype=np.float32)
            return np.clip(action.reshape(-1)[: self.config.action_dim], -1.0, 1.0)
        # Fallback: Phase 1 static default. Resolves via the SandboxAttack helper
        # so an externally supplied ``ctx.attacker_action`` still wins (preserves
        # the C2 discriminated-default contract).
        resolved = self.resolve_action(ctx, None, default_action=self.default_action)
        if resolved is None:
            resolved = np.asarray(self.default_action, dtype=np.float32)
        return np.asarray(resolved, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------ crafting

    def _craft_sybil_broadcast(self, ctx, decoded: BackdoorAction, num_attackers: int) -> List[Weights]:
        train_ctx = copy.copy(ctx)
        train_ctx.lr = decoded.local_lr
        train_ctx.local_epochs = decoded.local_epochs

        trained_models: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            loader = self._poison_loader(ctx, attacker_id, decoded.poison_grid_index)
            if loader is None:
                continue
            trained_models.append(train_on_loader(train_ctx, loader))
        if not trained_models:
            return self.fallback_old_weights(ctx)

        averaged = [
            np.mean([model[layer] for model in trained_models], axis=0)
            for layer in range(len(ctx.old_weights))
        ]
        crafted = [
            old + decoded.boost * (new - old)
            for old, new in zip(ctx.old_weights, averaged)
        ]
        if self.stealth_norm_cap:
            crafted = self._match_benign_norm(ctx, crafted)
        return [[layer.copy() for layer in crafted] for _ in range(num_attackers)]

    def _poison_loader(self, ctx, attacker_id: int, grid_index: int):
        grid = self.poisoned_train_iters(ctx).get("global_grid_by_attacker", {}).get(attacker_id)
        if grid:
            return grid[int(np.clip(grid_index, 0, len(grid) - 1))]
        return self.global_poisoned_loader_for_attacker(ctx, attacker_id)

    def _match_benign_norm(self, ctx, weights: Weights) -> Weights:
        defense_type = str(getattr(ctx, "defense_type", "fedavg")).lower()
        if defense_type == "fedavg":
            return weights
        benign_weights = getattr(ctx, "benign_weights", None) or []
        if not benign_weights:
            return weights
        benign_norm = float(np.mean([update_norm(ctx.old_weights, w) for w in benign_weights]))
        malicious_norm = update_norm(ctx.old_weights, weights)
        if benign_norm <= 0.0 or malicious_norm <= benign_norm or malicious_norm <= 1e-12:
            return weights
        scale = benign_norm / malicious_norm
        return [old + scale * (new - old) for old, new in zip(ctx.old_weights, weights)]

    # --------------------------------------------------------- observation

    def _build_live_observation(self, ctx) -> np.ndarray:
        previous = self._previous_global_weights or ctx.old_weights
        sampled_attacker_count = self.selected_attacker_count(ctx)
        sampled_client_count = max(
            sampled_attacker_count,
            int(_fl_value(ctx.fl_config, "fl", "num_clients", 1) * _fl_value(ctx.fl_config, "fl", "subsample_rate", 1.0)),
        )
        num_attackers_total = max(1, int(_fl_value(ctx.fl_config, "fl", "num_attackers", 1) or 1))
        return self._observation_builder.build(
            weights=ctx.old_weights,
            previous_weights=previous,
            last_action=self._last_action,
            round_idx=int(ctx.round_idx),
            sampled_attacker_count=int(sampled_attacker_count),
            num_attackers=num_attackers_total,
            sampled_client_count=int(sampled_client_count),
            local_bd_success=self._last_local_bd_success,
        )

    def _evaluate_local_bd_success(self, ctx, weights) -> float:
        loader = self._trigger_eval_loader(ctx)
        if loader is None or self._model_template is None or self._device is None or torch is None:
            return 0.0
        model = copy.deepcopy(self._model_template).to(self._device)
        set_model_weights(model, weights, self._device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                pred = model(images).argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.size(0))
        return float(correct) / max(1, total)

    def _trigger_eval_loader(self, ctx):
        iters = self.poisoned_train_iters(ctx)
        by_attacker = iters.get("trigger_eval_by_attacker", {}) or {}
        if not by_attacker:
            return None
        # Use the first eligible attacker — colluding sybils share the trigger.
        for attacker_id in sorted(by_attacker):
            return by_attacker[attacker_id]
        return None

    # ----------------------------------------------------------- simulator

    def _ensure_simulator(self, ctx) -> bool:
        if self._simulator is not None:
            self._simulator.initial_weights = [layer.copy() for layer in ctx.old_weights]
            return True
        if self._model_template is None or self._device is None:
            return False
        iters = self.poisoned_train_iters(ctx)
        grid_by_atk: Dict[int, list] = iters.get("global_grid_by_attacker", {}) or {}
        trigger_by_atk: Dict[int, Any] = iters.get("trigger_eval_by_attacker", {}) or {}
        if not grid_by_atk or not trigger_by_atk:
            return False
        attacker_id = sorted(grid_by_atk)[0]
        malicious_grid = grid_by_atk[attacker_id]
        trigger_eval = trigger_by_atk.get(attacker_id)
        if trigger_eval is None:
            return False

        attacker_loaders = ctx.selected_attacker_train_loaders or {}
        shadow_loaders = self._build_shadow_benign_loaders(attacker_loaders)
        if not shadow_loaders:
            # Coarse fallback: reuse the existing attacker loaders directly,
            # or a rate-0 view of the poisoned grid if none are available.
            shadow_loaders = (
                list(attacker_loaders.values()) if attacker_loaders else [malicious_grid[0]]
            )
        if not shadow_loaders:
            return False
        # Stealth-mode clean-accuracy reward needs a loader yielding (image,
        # TRUE_label) pairs that the attacker can run forward locally. The
        # first shadow shard fits that contract (it draws from the regular
        # train dataset, not the poisoned blend) and is already small enough
        # to forward in O(samples_per_client) per simulator step.
        clean_eval_loader = shadow_loaders[0]

        defender = AggregationDefender(
            defense_type=str(ctx.defense_type),
            krum_attackers=_fl_value(ctx.fl_config, "defender", "krum_attackers", 1),
            multi_krum_selected=_fl_value(ctx.fl_config, "defender", "multi_krum_selected", None),
            clipped_median_norm=_fl_value(ctx.fl_config, "defender", "clipped_median_norm", 2.0),
            trimmed_mean_ratio=_fl_value(ctx.fl_config, "defender", "trimmed_mean_ratio", 0.2),
            geometric_median_iters=_fl_value(ctx.fl_config, "defender", "geometric_median_iters", 10),
        )

        num_clients = max(2, int(_fl_value(ctx.fl_config, "fl", "num_clients", 4) or 4))
        num_attackers = max(1, int(_fl_value(ctx.fl_config, "fl", "num_attackers", 1) or 1))
        subsample = float(_fl_value(ctx.fl_config, "fl", "subsample_rate", 1.0) or 1.0)

        self._simulator = SimulatedBackdoorFLEnv(
            config=self.config,
            model_template=self._model_template,
            device=self._device,
            initial_weights=ctx.old_weights,
            benign_loader_factory=lambda cid: shadow_loaders[int(cid) % len(shadow_loaders)],
            malicious_grid=malicious_grid,
            trigger_eval_loader=trigger_eval,
            clean_eval_loader=clean_eval_loader,
            defender=defender,
            num_clients=num_clients,
            num_attackers=num_attackers,
            subsample_rate=subsample,
            seed=int(self.config.seed),
        )
        return True

    def _build_shadow_benign_loaders(self, attacker_loaders):
        """Sub-partition the attacker's local data into N distinct shadow shards.

        Old behaviour reused the few attacker DataLoaders directly for every
        simulated benign client (typically 2–20 loaders covering 50+ benign
        slots), which gave a degenerate benign-update distribution. With
        ``simulator_shadow_clients`` and ``simulator_shadow_samples_per_client``
        configured we deal the attacker's combined indices into independent
        shards — same underlying base dataset, disjoint index sets — so each
        simulated benign client trains on its own data slice.
        """
        if not attacker_loaders or self.config.simulator_shadow_clients <= 0:
            return None
        first = next(iter(attacker_loaders.values()))
        base_dataset = getattr(getattr(first, "dataset", None), "dataset", None)
        if base_dataset is None:
            return None
        from torch.utils.data import DataLoader

        from fl_sandbox.data import DatasetSplit

        all_idxs: list[int] = []
        for loader in attacker_loaders.values():
            split = getattr(loader, "dataset", None)
            if split is None:
                continue
            all_idxs.extend(int(idx) for idx in getattr(split, "idxs", []) or [])
        if not all_idxs:
            return None

        n_clients = int(self.config.simulator_shadow_clients)
        per_client = int(self.config.simulator_shadow_samples_per_client)
        rng = np.random.default_rng(int(self.config.seed))
        idxs = np.asarray(all_idxs, dtype=np.int64)
        rng.shuffle(idxs)
        needed = n_clients * per_client
        if needed > len(idxs):
            # Top up with sampling-with-replacement so shards still get
            # ``per_client`` samples each — disjoint where possible.
            extra = rng.choice(idxs, size=needed - len(idxs), replace=True)
            idxs = np.concatenate([idxs, extra])

        batch_size = int(getattr(first, "batch_size", 32) or 32)
        loaders = []
        for k in range(n_clients):
            shard = idxs[k * per_client : (k + 1) * per_client].tolist()
            if not shard:
                continue
            loaders.append(
                DataLoader(
                    DatasetSplit(base_dataset, shard),
                    batch_size=min(batch_size, len(shard)),
                    shuffle=True,
                )
            )
        return loaders or None

    def _train_policy_one_round(self) -> None:
        if self._simulator is None:
            return
        if self._trainer is None:
            self._trainer = build_trainer(self.config)
        steps = max(1, int(self.policy_train_steps_per_round))
        started = time.perf_counter()
        collect = self._trainer.collect(self._simulator, steps=steps)
        update = self._trainer.update(
            gradient_steps=max(1, steps // max(1, int(self.config.train_freq_steps)))
        )
        train_time = time.perf_counter() - started
        trainer_metrics = {}
        for key, value in self._trainer.diagnostics().items():
            if key in {"trainer_collect_steps", "trainer_update_steps"}:
                continue
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                trainer_metrics[f"rl_{key}"] = float(value)
        simulator_metrics = {
            f"rl_simulated_{key}": float(value)
            for key, value in collect.info_means.items()
            if isinstance(value, (int, float, np.number)) and np.isfinite(value)
        }
        self._diagnostics.update(
            {
                **trainer_metrics,
                **simulator_metrics,
                "rl_trainer_collect_steps": float(collect.steps),
                "rl_trainer_update_steps": float(update.gradient_steps),
                "rl_simulated_reward": float(collect.reward_mean),
                "rl_trainer_last_update_loss": float(update.loss),
                "rl_trainer_train_time": float(train_time),
            }
        )

    def _maybe_load_checkpoint(self, round_idx: int) -> None:
        if self._checkpoint_loaded or self.config is None:
            return
        path = self.config.checkpoint_for_round(int(round_idx))
        if not path:
            return
        if self._trainer is None:
            self._trainer = build_trainer(self.config)
        # Materialise the policy network from config shapes alone so checkpoint
        # load works in pure-deployment runs (no simulator built).
        import gymnasium as gym
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.observation_dim,), dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=self.config.action_low, high=self.config.action_high, dtype=np.float32,
        )
        self._trainer.ensure_initialized(obs_space, action_space)
        try:
            self._trainer.load(path)
            self._checkpoint_loaded = True
            self._diagnostics["rl_backdoor_checkpoint_loaded"] = 1.0
        except (FileNotFoundError, RuntimeError):
            self._diagnostics["rl_backdoor_checkpoint_loaded"] = 0.0
