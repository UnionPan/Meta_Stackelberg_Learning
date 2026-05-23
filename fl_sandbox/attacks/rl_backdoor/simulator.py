"""Grey-box simulator for RL backdoor policy training.

Mirrors the structural rules of an FL round — sample clients, train locally,
defender aggregates — but with attacker-owned shadow data only. The RL policy
trains here for many episodes without ever touching real FL; this is what
makes the threat model honest (the deployed attacker cannot run thousands of
real FL rounds) and what lets us measure a sim→real transfer gap.

Hard constraints:

- The shadow head MUST share the deployed model's architecture, otherwise
  the projected tail-layer state has a different layout and the trained
  policy will not transfer.
- Every observation component must be computable identically in this
  simulator and in the live attacker. The same ``BackdoorObservationBuilder``
  is reused on both sides.
- Privileged global metrics never feed the observation. ``poi_acc`` is a
  shadow trigger-set evaluation — it does NOT represent the true global ASR
  and is not exposed to the policy state.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

from fl_sandbox.attacks.base import get_model_weights, set_model_weights
from fl_sandbox.attacks.rl_backdoor.action import decode_backdoor_action
from fl_sandbox.attacks.rl_backdoor.config import BackdoorRLConfig
from fl_sandbox.attacks.rl_backdoor.observation import BackdoorObservationBuilder


class _Box:
    def __init__(self, *, low, high, shape=None, dtype=np.float32) -> None:
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = tuple(shape or self.low.shape)
        self.dtype = dtype


def _box(*, low, high, shape=None, dtype=np.float32):
    if gym is not None:
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _Box(low=low, high=high, shape=shape, dtype=dtype)


class SimulatedBackdoorFLEnv:
    """Gym-style env that simulates one FL round with attacker-owned data.

    Built lazily from primitives the attacker can construct in live FL:

    - ``model_template`` — same architecture as the deployed FL model.
    - ``initial_weights`` — current global snapshot (taken in ``observe_round``).
    - ``benign_loader_factory(client_id)`` — produces a shadow benign client's
      loader (e.g. attacker's data partitioned into shards).
    - ``malicious_grid`` — list of 11 fully-poisoned shadow loaders at trigger
      rates 0.0..1.0, indexed by ``decode_backdoor_action(action).poison_grid_index``.
    - ``trigger_eval_loader`` — small loader of triggered samples whose targets
      are the attack's target class; used for the paper backdoor loss and for
      the ``local_bd_success`` observation component.
    - ``defender`` — same ``AggregationDefender`` API as the live runner.

    Episode length is ``config.train_horizon``; the horizon ends with
    ``truncated=True, terminated=False`` (no absorbing state in FL — closes C5).
    """

    def __init__(
        self,
        *,
        config: BackdoorRLConfig,
        model_template,
        device,
        initial_weights,
        benign_loader_factory: Callable[[int], object],
        malicious_grid: List[object],
        trigger_eval_loader,
        defender,
        num_clients: int,
        num_attackers: int,
        subsample_rate: float = 1.0,
        seed: int = 0,
        clean_eval_loader=None,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for SimulatedBackdoorFLEnv")
        self.config = config
        self.model_template = model_template
        self.device = device
        self.initial_weights: List[np.ndarray] = [layer.copy() for layer in initial_weights]
        self.benign_loader_factory = benign_loader_factory
        self.malicious_grid = list(malicious_grid)
        self.trigger_eval_loader = trigger_eval_loader
        # Stealth reward needs a clean-accuracy estimate the attacker can
        # actually compute on its own data. If not provided, stealth-mode
        # ``clean_drop`` collapses to 0 (paper mode stays unaffected).
        self.clean_eval_loader = clean_eval_loader
        self.defender = defender
        self.num_clients = int(num_clients)
        self.num_attackers = int(num_attackers)
        self.subsample_rate = float(subsample_rate)
        self.rng = np.random.default_rng(int(seed))

        self.observation_builder = BackdoorObservationBuilder(self.config)
        sample_obs = self._compose_observation(
            weights=self.initial_weights,
            previous_weights=self.initial_weights,
            last_action=np.zeros(self.config.action_dim, dtype=np.float32),
            round_idx=0,
            sampled_attacker_count=0,
            sampled_client_count=max(1, int(self.num_clients * self.subsample_rate)),
            local_bd_success=0.0,
        )
        self.observation_builder.reset()
        self.observation_space = _box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )
        self.action_space = _box(
            low=self.config.action_low, high=self.config.action_high, dtype=np.float32
        )

        self.current_weights: List[np.ndarray] = []
        self.previous_weights: List[np.ndarray] = []
        self.round_idx: int = 0
        self.last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.last_poi_acc: float = 0.0
        self.last_clean_acc: float = 0.0

    def reset(self, *, seed: Optional[int] = None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.current_weights = [layer.copy() for layer in self.initial_weights]
        self.previous_weights = [layer.copy() for layer in self.initial_weights]
        self.round_idx = 0
        self.last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.last_poi_acc = float(self._evaluate_poi_acc(self.current_weights))
        self.last_clean_acc = float(self._evaluate_clean_acc(self.current_weights))
        self.observation_builder.reset()
        obs = self._compose_observation(
            weights=self.current_weights,
            previous_weights=self.previous_weights,
            last_action=self.last_action,
            round_idx=self.round_idx,
            sampled_attacker_count=0,
            sampled_client_count=max(1, int(self.num_clients * self.subsample_rate)),
            local_bd_success=self.last_poi_acc,
        )
        return obs, {"round_idx": self.round_idx, "poi_acc": self.last_poi_acc}

    def step(self, action):
        if not self.current_weights:
            raise RuntimeError("SimulatedBackdoorFLEnv.step called before reset()")
        raw = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1)[: self.config.action_dim],
            self.config.action_low,
            self.config.action_high,
        ).astype(np.float32)
        decoded = decode_backdoor_action(raw)

        self.round_idx += 1
        sampled_clients, sampled_attackers = self._sample_clients()
        sampled_attacker_count = len(sampled_attackers)
        sampled_client_count = max(1, len(sampled_clients))

        benign_weights = [
            self._train_benign(cid)
            for cid in sampled_clients
            if cid not in sampled_attackers
        ]
        all_weights = list(benign_weights)
        mal: Optional[List[np.ndarray]] = None
        if sampled_attackers:
            mal = self._craft_malicious(decoded)
            all_weights.extend([[layer.copy() for layer in mal] for _ in sampled_attackers])

        old_weights = [layer.copy() for layer in self.current_weights]
        self.previous_weights = old_weights
        if all_weights:
            self.current_weights = self.defender.aggregate(old_weights, all_weights)

        poi_acc, backdoor_loss = self._classifier_metrics(
            self.current_weights, self.trigger_eval_loader
        )
        clean_acc, clean_loss = self._classifier_metrics(
            self.current_weights, self.clean_eval_loader
        )
        mal_norm = self._update_norm(old_weights, mal) if mal is not None else 0.0
        benign_norm = (
            float(np.mean([self._update_norm(old_weights, w) for w in benign_weights]))
            if benign_weights else 0.0
        )
        attack_objective = self._attack_objective(
            clean_loss=clean_loss,
            backdoor_loss=backdoor_loss,
        )
        reward = self._reward(
            poi_acc=poi_acc,
            prev_poi_acc=self.last_poi_acc,
            clean_acc=clean_acc,
            prev_clean_acc=self.last_clean_acc,
            mal_norm=mal_norm,
            benign_norm=benign_norm,
            clean_loss=clean_loss,
            backdoor_loss=backdoor_loss,
            sampled_attacker_count=sampled_attacker_count,
        )
        self.last_poi_acc = poi_acc
        self.last_clean_acc = clean_acc
        self.last_action = raw

        obs = self._compose_observation(
            weights=self.current_weights,
            previous_weights=self.previous_weights,
            last_action=self.last_action,
            round_idx=self.round_idx,
            sampled_attacker_count=sampled_attacker_count,
            sampled_client_count=sampled_client_count,
            local_bd_success=poi_acc,
        )
        truncated = self.round_idx >= max(1, int(self.config.train_horizon))
        info = {
            "round_idx": self.round_idx,
            "poi_acc": poi_acc,
            "clean_acc": clean_acc,
            "clean_loss": clean_loss,
            "backdoor_loss": backdoor_loss,
            "attack_objective": attack_objective,
            "mal_norm": mal_norm,
            "benign_norm": benign_norm,
            "sampled_attackers": sampled_attacker_count,
            "sampled_clients": sampled_client_count,
            "decoded_poison_grid_index": int(decoded.poison_grid_index),
            "decoded_boost": float(decoded.boost),
        }
        return obs, float(reward), False, bool(truncated), info

    def _compose_observation(self, **kwargs) -> np.ndarray:
        return self.observation_builder.build(num_attackers=self.num_attackers, **kwargs)

    def _sample_clients(self) -> tuple[list[int], set[int]]:
        n_sample = max(1, int(self.num_clients * self.subsample_rate))
        n_sample = min(n_sample, self.num_clients)
        sampled = sorted(
            self.rng.choice(self.num_clients, size=n_sample, replace=False).tolist()
        )
        attacker_ids = set(range(self.num_attackers))
        return sampled, {cid for cid in sampled if cid in attacker_ids}

    def _train_benign(self, client_id: int):
        loader = self.benign_loader_factory(int(client_id))
        return self._sgd(loader, self.current_weights)

    def _craft_malicious(self, decoded):
        grid_idx = int(np.clip(decoded.poison_grid_index, 0, len(self.malicious_grid) - 1))
        loader = self.malicious_grid[grid_idx]
        trained = self._sgd(
            loader,
            self.current_weights,
            lr_override=decoded.local_lr,
            epochs_override=decoded.local_epochs,
        )
        return [
            old + decoded.boost * (new - old)
            for old, new in zip(self.current_weights, trained)
        ]

    def _sgd(
        self,
        loader,
        weights,
        *,
        lr_override: Optional[float] = None,
        epochs_override: Optional[int] = None,
    ):
        model = copy.deepcopy(self.model_template).to(self.device)
        set_model_weights(model, weights, self.device)
        criterion = torch.nn.CrossEntropyLoss()
        lr = float(lr_override if lr_override is not None else self.config.simulator_lr)
        epochs = int(
            epochs_override
            if epochs_override is not None
            else self.config.simulator_local_epochs
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=max(1e-6, lr))
        model.train()
        for _ in range(max(1, epochs)):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        return get_model_weights(model)

    def _evaluate_poi_acc(self, weights) -> float:
        return self._classifier_metrics(weights, self.trigger_eval_loader)[0]

    def _evaluate_clean_acc(self, weights) -> float:
        return self._classifier_metrics(weights, self.clean_eval_loader)[0]

    def _classifier_accuracy(self, weights, loader) -> float:
        return self._classifier_metrics(weights, loader)[0]

    def _classifier_metrics(self, weights, loader) -> tuple[float, float]:
        if loader is None:
            return 0.0, 0.0
        model = copy.deepcopy(self.model_template).to(self.device)
        set_model_weights(model, weights, self.device)
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = model(images)
                pred = logits.argmax(dim=1)
                batch_size = int(labels.size(0))
                correct += int((pred == labels).sum().item())
                total += batch_size
                loss_sum += float(criterion(logits, labels).item()) * batch_size
        return float(correct) / max(1, total), float(loss_sum) / max(1, total)

    @staticmethod
    def _update_norm(old, new) -> float:
        if new is None:
            return 0.0
        scaled_sq = 0.0
        max_abs = 0.0
        for old_layer, new_layer in zip(old, new):
            diff = np.asarray(new_layer, dtype=np.float64).reshape(-1) - np.asarray(
                old_layer, dtype=np.float64
            ).reshape(-1)
            layer_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            if layer_max <= 0.0:
                continue
            if layer_max > max_abs and max_abs > 0.0:
                scaled_sq *= (max_abs / layer_max) ** 2
                max_abs = layer_max
            elif max_abs == 0.0:
                max_abs = layer_max
            scaled = diff / max_abs
            scaled_sq += float(np.dot(scaled, scaled))
        if max_abs == 0.0:
            return 0.0
        return float(max_abs * np.sqrt(scaled_sq))

    def _reward(
        self,
        *,
        poi_acc: float,
        prev_poi_acc: float,
        clean_acc: float,
        prev_clean_acc: float,
        mal_norm: float,
        benign_norm: float,
        clean_loss: float = 0.0,
        backdoor_loss: float = 0.0,
        sampled_attacker_count: int = 1,
    ) -> float:
        mode = str(self.config.reward_mode).lower()
        gain = float(poi_acc - prev_poi_acc)
        if mode in {"paper", "henger_li", "henger-li"}:
            if int(sampled_attacker_count) <= 0:
                return 0.0
            return -self._attack_objective(
                clean_loss=clean_loss,
                backdoor_loss=backdoor_loss,
            )
        if mode in {"delta", "asr_delta", "poi_delta"}:
            return gain
        # Stealth: penalise clean-accuracy drops and norm outliers. Both signals
        # are observable to the attacker via its own shadow data (no privileged
        # global metrics consulted). gamma must drop below 1 when this mode is
        # active — the penalties are unbounded below and would break the
        # critic's bounded-value-function assumption under gamma=1.
        clean_drop = max(0.0, float(prev_clean_acc) - float(clean_acc))
        norm_excess = (
            max(0.0, float(mal_norm) / float(benign_norm) - 1.0)
            if benign_norm > 1e-12 else 0.0
        )
        return (
            gain
            - float(self.config.reward_clean_weight) * clean_drop
            - float(self.config.reward_norm_weight) * norm_excess
        )

    def _attack_objective(self, *, clean_loss: float, backdoor_loss: float) -> float:
        clean_lambda = float(np.clip(self.config.reward_clean_lambda, 0.0, 1.0))
        return (
            clean_lambda * float(clean_loss)
            + (1.0 - clean_lambda) * float(backdoor_loss)
        )
