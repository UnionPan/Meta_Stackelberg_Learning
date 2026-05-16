"""Adapter from meta_sg's FLCoordinator interface to fl_sandbox."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np

from fl_sandbox.attacks import create_attack
from fl_sandbox.federation.runner import MinimalFLRunner, SandboxConfig
from fl_sandbox.utils import set_parameters

from meta_sg.simulation.interface import FLCoordinator
from meta_sg.simulation.types import InitialState, RoundSummary, SimulationSnapshot, SimulationSpec, Weights


class FLSandboxCoordinatorAdapter(FLCoordinator):
    """
    Use ``fl_sandbox.federation.runner.MinimalFLRunner`` as a real FL backend.

    This adapter keeps the meta_sg interface paper-shaped:
    - BSMGEnv still emits one FL round per ``step``.
    - Per-round ``DefenseDecision(alpha, beta, epsilon/sigma)`` is passed into
      fl_sandbox as norm-bound + trimmed-mean aggregation.
    - Post-training defenses remain outside the FL transition.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self.runner = MinimalFLRunner(self.config)
        self._round_idx = 0
        self._last_summary: Optional[RoundSummary] = None

    def reset(self, seed: Optional[int] = None) -> InitialState:
        if seed is not None and seed != self.config.seed:
            self.config.seed = int(seed)
            self.runner = MinimalFLRunner(self.config)
        else:
            self.runner.reset_model()
        self._round_idx = 0
        self._last_summary = None
        return InitialState(weights=self.current_weights, round_idx=0)

    def run_round(
        self,
        attack,
        defense,
        attack_decision,
        defense_decision,
        evaluate: bool = True,
    ) -> RoundSummary:
        self._round_idx += 1
        attack_name = _attack_name(attack)
        sandbox_attack = self._build_attack(attack_name, attack_decision)
        should_evaluate = evaluate or self._last_summary is None
        summary = self.runner.run_round(
            self._round_idx,
            attack=sandbox_attack,
            evaluate=should_evaluate,
            attacker_action=getattr(attack_decision, "raw", None),
            defense_decision=defense_decision,
        )
        translated = self._translate_summary(summary)
        if should_evaluate:
            self._last_summary = translated
        return translated

    @property
    def current_weights(self) -> Weights:
        return [w.copy() for w in self.runner.current_weights]

    @property
    def spec(self) -> SimulationSpec:
        return SimulationSpec(layer_shapes=tuple(tuple(w.shape) for w in self.runner.current_weights))

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            round_idx=self._round_idx,
            weights=self.current_weights,
            rng_state=None,
        )

    def restore(self, snapshot: SimulationSnapshot) -> None:
        self._round_idx = snapshot.round_idx
        self.runner.current_weights = [w.copy() for w in snapshot.weights]
        set_parameters(self.runner.model, self.runner.current_weights)

    def _build_attack(self, attack_name: str, attack_decision):
        cfg = _attack_config_from_sandbox(self.config, attack_name, attack_decision)
        return create_attack(cfg)

    def _translate_summary(self, summary) -> RoundSummary:
        clean_acc = _finite_or_cached(summary.clean_acc, self._last_summary.clean_acc if self._last_summary else 0.0)
        backdoor_acc = _finite_or_cached(
            summary.backdoor_acc,
            self._last_summary.backdoor_acc if self._last_summary else 0.0,
        )
        clean_loss = _finite_or_cached(summary.clean_loss, self._last_summary.clean_loss if self._last_summary else 0.0)
        return RoundSummary(
            round_idx=summary.round_idx,
            clean_acc=clean_acc,
            backdoor_acc=backdoor_acc,
            clean_loss=clean_loss,
            attack_name=summary.attack_name,
            defense_name=summary.defense_name,
            benign_update_norms=list(getattr(summary, "benign_update_norms", [])),
            malicious_update_norms=list(getattr(summary, "malicious_update_norms", [])),
            malicious_cosines_to_benign=list(getattr(summary, "malicious_cosines_to_benign", [])),
            selected_attackers=list(getattr(summary, "selected_attackers", [])),
            sampled_clients=list(getattr(summary, "sampled_clients", [])),
        )


def _attack_name(attack) -> str:
    if attack is None:
        return "clean"
    attack_type = getattr(attack, "attack_type", None)
    if attack_type is not None:
        return str(getattr(attack_type, "name", "clean")).lower()
    return str(getattr(attack, "name", "clean")).lower()


def _attack_config_from_sandbox(config: SandboxConfig, attack_name: str, attack_decision) -> SimpleNamespace:
    values = dict(vars(config))
    values.update(
        {
            "type": attack_name,
            "alie_tau": getattr(config, "alie_tau", 1.0),
            "gaussian_sigma": getattr(config, "gaussian_sigma", 0.1),
            "attacker_action": tuple(np.asarray(getattr(attack_decision, "raw", config.attacker_action), dtype=float)),
        }
    )
    return SimpleNamespace(**values)


def _finite_or_cached(value: float, fallback: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)
