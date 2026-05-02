"""Core strategy types: AttackType, DefenseDecision, AttackDecision."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AttackType:
    """
    Identifies one attack scenario ξ = {(ω1, ω2, ω3)_i}^M_i.
    Corresponds to Section II-D of the paper.
    """
    name: str                    # "ipm" | "lmp" | "bfl" | "dba" | "rl" | "brl"
    objective: str               # "untargeted" | "targeted"
    adaptive: bool               # True = RL attacker with inner best-response loop
    config: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}({'adaptive' if self.adaptive else 'fixed'})"


# Pre-defined attack domain Ξ from paper (Table 3)
ATTACK_DOMAIN = {
    "ipm": AttackType(name="ipm", objective="untargeted", adaptive=False,
                      config={"scaling": 2.0}),
    "lmp": AttackType(name="lmp", objective="untargeted", adaptive=False,
                      config={"scale": 5.0}),
    "bfl": AttackType(name="bfl", objective="targeted", adaptive=False,
                      config={"poison_frac": 1.0}),
    "dba": AttackType(name="dba", objective="targeted", adaptive=False,
                      config={"poison_frac": 0.5, "num_sub_triggers": 4}),
    "rl":  AttackType(name="rl",  objective="untargeted", adaptive=True),
    "brl": AttackType(name="brl", objective="targeted",   adaptive=True),
}


@dataclass(frozen=True)
class DefenseDecision:
    """
    Paper defender action a_D = (α, β, ε/σ).

    α: norm bound — clip all client updates to L2 norm ≤ α
    β: trimmed mean ratio — discard β fraction from each tail per coordinate
    neuroclip_epsilon: NeuroClip clip range (post-training, reward only)
    prun_mask_rate: Pruning mask rate (post-training, reward only)
    """
    norm_bound_alpha: float
    trimmed_mean_beta: float
    neuroclip_epsilon: Optional[float] = None
    prun_mask_rate: Optional[float] = None

    @classmethod
    def from_raw(
        cls,
        raw: np.ndarray,
        alpha_max: float = 5.0,
        beta_max: float = 0.45,
        eps_max: float = 10.0,
        use_neuroclip: bool = True,
    ) -> "DefenseDecision":
        """Decode raw ∈ [-1, 1]^3 action to physical defense parameters."""
        a = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
        alpha = float((a[0] + 1) / 2 * alpha_max)
        beta = float((a[1] + 1) / 2 * beta_max)
        post = float((a[2] + 1) / 2 * eps_max)
        if use_neuroclip:
            return cls(norm_bound_alpha=alpha, trimmed_mean_beta=beta,
                       neuroclip_epsilon=max(1.0, post))
        return cls(norm_bound_alpha=alpha, trimmed_mean_beta=beta,
                   prun_mask_rate=float(np.clip(post / eps_max, 0.0, 0.5)))


@dataclass
class AttackDecision:
    """
    Decoded attacker action — 3D continuous space after compression.
    Paper §Appendix C: 3-dimensional real action space after compression.
    """
    raw: np.ndarray
    gamma_scale: float = 1.5     # scaling for malicious update magnitude
    local_steps: int = 10        # local gradient steps
    lambda_stealth: float = 0.5  # stealth regularisation weight

    @classmethod
    def from_raw(cls, raw: np.ndarray) -> "AttackDecision":
        """Decode raw ∈ [-1, 1]^3 to attacker parameters (Table in td3_attacker)."""
        a = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
        if a.shape[0] < 3:
            a = np.pad(a, (0, 3 - a.shape[0]))
        return cls(
            raw=a.copy(),
            gamma_scale=max(0.1, float(a[0]) * 1.4 + 1.5),
            local_steps=max(1, int(round(float(a[1]) * 9.0 + 10.0))),
            lambda_stealth=float(np.clip(float(a[2]) * 0.45 + 0.5, 0.0, 1.0)),
        )
