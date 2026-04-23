"""Backdoor-style attacker implementations.

Main APIs in this file:
- ``BFLAttack``
- ``DBAAttack``
- ``BRLAttack``
- ``SelfGuidedBRLAttack``
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from .base import (
    SandboxAttack,
    Weights,
    bounded_boost,
    bounded_local_epochs,
    bounded_local_lr,
    get_model_weights,
    set_model_weights,
    train_on_loader,
)
from fl_sandbox.core.metrics import update_norm


@dataclass
class BFLAttack(SandboxAttack):
    """Backdoor FL using a single global trigger pattern."""

    poison_frac: float = 1.0
    name: str = "BFL"
    attack_type: str = "bfl"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []
        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            poisoned_loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if poisoned_loader is None:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            malicious_weights.append(train_on_loader(ctx, poisoned_loader))
        return malicious_weights


@dataclass
class DBAAttack(SandboxAttack):
    """Distributed Backdoor Attack using one sub-trigger per attacker."""

    num_sub_triggers: int = 4
    poison_frac: float = 0.5
    name: str = "DBA"
    attack_type: str = "dba"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []

        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            sub_loaders = self.sub_trigger_loaders_for_attacker(ctx, attacker_id)
            if not sub_loaders:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            max_subs = min(self.num_sub_triggers, len(sub_loaders))
            sub_idx = attacker_id % max_subs
            malicious_weights.append(train_on_loader(ctx, sub_loaders[sub_idx]))
        return malicious_weights


@dataclass
class BRLAttack(SandboxAttack):
    """Adaptive backdoor attack aligned with the reference paper implementation."""

    default_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "BRL"
    attack_type: str = "brl"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []

        action = self.resolve_action(ctx, attacker_action, default_action=self.default_action)
        if action is None:
            return self.fallback_old_weights(ctx)

        train_ctx = copy.copy(ctx)
        train_ctx.lr = bounded_local_lr(float(action[1]))
        train_ctx.local_epochs = bounded_local_epochs(float(action[2]))
        boost = bounded_boost(float(action[0]))

        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            # Each attacker trains only on their own local poisoned slice — using the
            # union of all attackers' data would assume colluding data sharing, which
            # is a stronger (and unfair) threat model compared to BFL/DBA.
            loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if loader is None:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            trained = train_on_loader(train_ctx, loader)
            # Boost amplifies the malicious delta: boost=1 → submit as-is, boost>1 → amplify
            boosted = [o + boost * (t - o) for o, t in zip(ctx.old_weights, trained)]
            malicious_weights.append(boosted)
        return malicious_weights


# ---------------------------------------------------------------------------
# Self-guided BRL: internal TD3 policy, no external action required
# ---------------------------------------------------------------------------

_ACTION_LOW  = np.array([-1., -1., -1.], dtype=np.float32)
_ACTION_HIGH = np.array([ 1.,  1.,  1.], dtype=np.float32)


@dataclass
class SelfGuidedBRLAttack(SandboxAttack):
    """Backdoor attacker with an internal TD3 policy.

    Learns to tune (boost, local_lr, local_epochs) each round to maximise
    ASR while penalising updates whose norm exceeds the benign mean (stealth).

    Unlike ``BRLAttack``, which relies on an externally-supplied action, this
    attacker manages its own RL loop:

    * **State** — compressed tail weights of the global model + normalised
      round index.
    * **Action** — (boost_raw, lr_raw, epoch_raw) ∈ [-1, 1]³, decoded via
      the same ``bounded_*`` helpers used by BRLAttack.
    * **Reward** — ``-CE_loss(model, poisoned_batch)``
      ``- stealth_lambda * max(0, mal_norm - mean_benign_norm)``
    * **Policy** — TD3 (shared implementation from ``rl.attacker``).
    """

    stealth_lambda: float = 0.5
    attack_start_round: int = 5
    exploration_noise: float = 0.2
    td3_batch_size: int = 64
    replay_capacity: int = 10_000
    policy_lr: float = 3e-4
    num_tail_layers: int = 2
    name: str = "SelfGuidedBRL"
    attack_type: str = "sgbrl"

    # --- internal state (not dataclass fields, set in __post_init__) ---
    def __post_init__(self) -> None:
        self._policy = None          # TD3Agent, lazy-init on first round
        self._replay = None          # ReplayBuffer
        self._model_template = None  # cpu copy of the global model
        self._device = torch.device("cpu") if torch is not None else None
        # carried across rounds for the TD3 transition
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        self._prev_mal_norm: float = 0.0
        # cached so execute() doesn't recompute it
        self._current_state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _build_state(self, ctx) -> np.ndarray:
        """Compress global model weights + normalised round index."""
        from src.models.cnn import get_compressed_state
        if self._model_template is None or torch is None:
            return np.zeros(1, dtype=np.float32)
        model = copy.deepcopy(self._model_template).to(self._device)
        set_model_weights(model, ctx.old_weights, self._device)
        compressed, _ = get_compressed_state(model, num_tail_layers=self.num_tail_layers)
        round_norm = np.array(
            [2.0 * min(ctx.round_idx, 200) / 200.0 - 1.0], dtype=np.float32
        )
        return np.concatenate([compressed.astype(np.float32), round_norm])

    def _state_dim(self, ctx) -> int:
        return self._build_state(ctx).shape[0]

    # ------------------------------------------------------------------
    # Policy initialisation
    # ------------------------------------------------------------------

    def _ensure_policy(self, ctx) -> None:
        if self._policy is not None:
            return
        from fl_sandbox.core.rl.attacker import RLAttackerConfig, ReplayBuffer, TD3Agent
        state_dim = self._state_dim(ctx)
        cfg = RLAttackerConfig(
            policy_lr=self.policy_lr,
            td3_batch_size=self.td3_batch_size,
        )
        self._policy = TD3Agent(state_dim, _ACTION_LOW, _ACTION_HIGH, cfg, self._device)
        self._replay = ReplayBuffer(capacity=self.replay_capacity)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, ctx) -> float:
        """Evaluate how well the backdoor persisted after the defender aggregated."""
        if self._model_template is None or torch is None:
            return 0.0

        # find any available local poisoned loader for the reward estimate
        loader = None
        for attacker_id in ctx.selected_attacker_ids:
            loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if loader is not None:
                break
        # fall back to global poisoned loader if per-attacker not available
        if loader is None:
            loader = self.global_poisoned_loader(ctx)
        if loader is None:
            return 0.0

        model = copy.deepcopy(self._model_template).to(self._device)
        set_model_weights(model, ctx.old_weights, self._device)
        model.eval()
        images, labels = next(iter(loader))
        images = images.to(self._device)
        labels = labels.to(self._device)
        with torch.no_grad():
            loss = F.cross_entropy(model(images), labels)
        asr_reward = -float(loss.item())

        # stealth: penalise if previous malicious norm exceeded benign mean
        stealth_penalty = 0.0
        if ctx.benign_weights and self._prev_mal_norm > 0.0:
            benign_norms = [update_norm(ctx.old_weights, w) for w in ctx.benign_weights]
            mean_benign = float(np.mean(benign_norms)) if benign_norms else 0.0
            stealth_penalty = max(0.0, self._prev_mal_norm - mean_benign)

        return asr_reward - self.stealth_lambda * stealth_penalty

    # ------------------------------------------------------------------
    # SandboxAttack interface
    # ------------------------------------------------------------------

    def observe_round(self, ctx) -> None:
        if torch is None or ctx.model is None or ctx.device is None:
            return

        # lazy init model template and device
        if self._model_template is None:
            self._model_template = copy.deepcopy(ctx.model).cpu()
        self._device = ctx.device

        # build current state (cached for execute())
        self._current_state = self._build_state(ctx)

        # close the previous transition now that we can measure the reward
        if self._prev_state is not None and self._prev_action is not None:
            reward = self._compute_reward(ctx)
            self._ensure_policy(ctx)
            self._replay.add(
                self._prev_state,
                self._prev_action,
                reward,
                self._current_state,
                done=False,
            )
            self._policy.train_step(self._replay)

    def execute(self, ctx, attacker_action=None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []
        if torch is None:
            return self.fallback_old_weights(ctx)

        self._ensure_policy(ctx)

        # select action: random explore until warm-up, then policy
        state = self._current_state if self._current_state is not None else self._build_state(ctx)
        if ctx.round_idx < self.attack_start_round or len(self._replay) < self.td3_batch_size:
            action = np.random.uniform(_ACTION_LOW, _ACTION_HIGH).astype(np.float32)
        else:
            action = self._policy.act(state, noise_scale=self.exploration_noise)

        # store for next round's TD3 transition
        self._prev_state = state.copy()
        self._prev_action = action.copy()

        # decode action to training hyper-parameters
        boost = bounded_boost(float(action[0]))
        train_ctx = copy.copy(ctx)
        train_ctx.lr = bounded_local_lr(float(action[1]))
        train_ctx.local_epochs = bounded_local_epochs(float(action[2]))

        malicious_weights: List[Weights] = []
        for attacker_id in ctx.selected_attacker_ids:
            loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if loader is None:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            trained = train_on_loader(train_ctx, loader)
            boosted = [o + boost * (t - o) for o, t in zip(ctx.old_weights, trained)]
            malicious_weights.append(boosted)

        # record malicious norm for next round's stealth penalty
        if malicious_weights:
            mal_norms = [update_norm(ctx.old_weights, w) for w in malicious_weights]
            self._prev_mal_norm = float(np.mean(mal_norms))

        return malicious_weights
