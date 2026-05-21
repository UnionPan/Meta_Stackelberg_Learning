"""Paper-inspired RL/IL backdoor attacker."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import List, Optional

import numpy as np

from fl_sandbox.attacks.base import SandboxAttack, Weights, train_on_loader
from fl_sandbox.attacks.rl_backdoor.action import decode_backdoor_action
from fl_sandbox.core.metrics import update_norm


@dataclass
class RLBackdoorAttack(SandboxAttack):
    """Backdoor attacker controlled by one shared continuous action."""

    default_action: tuple[float, float, float, float] = (1.0, 0.0, -1.0, 0.0)
    name: str = "RLBackdoor"
    attack_type: str = "rl_backdoor"

    def execute(self, ctx, attacker_action: Optional[np.ndarray] = None) -> List[Weights]:
        if self.selected_attacker_count(ctx) == 0:
            return []

        raw_action = self.resolve_action(ctx, attacker_action, default_action=self.default_action)
        if raw_action is None:
            return self.fallback_old_weights(ctx)
        action = decode_backdoor_action(raw_action)

        train_ctx = copy.copy(ctx)
        train_ctx.lr = action.local_lr
        train_ctx.local_epochs = action.local_epochs

        malicious_weights: List[Weights] = []
        shared_boosted: Weights | None = None
        for attacker_id in ctx.selected_attacker_ids:
            loader = self.global_poisoned_loader_for_attacker(ctx, attacker_id)
            if loader is None:
                malicious_weights.append(self.clone_old_weights(ctx))
                continue
            trained = train_on_loader(train_ctx, loader)
            if shared_boosted is None:
                shared_boosted = [old + action.boost * (new - old) for old, new in zip(ctx.old_weights, trained)]
                shared_boosted = self._match_benign_norm(ctx, shared_boosted)
            malicious_weights.append([layer.copy() for layer in shared_boosted])
        return malicious_weights

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
