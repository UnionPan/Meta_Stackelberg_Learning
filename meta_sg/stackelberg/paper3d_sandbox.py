"""fl_sandbox bridge for paper-aligned three-dimensional defender actions."""

from __future__ import annotations

from src.defenses import apply_post_defense

from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter, _attack_name


class FrozenSandboxAttack:
    """Descriptor that makes FLSandboxCoordinatorAdapter build native fl_sandbox RLAttack."""

    name = "rl"


class Paper3DSandboxCoordinator(FLSandboxCoordinatorAdapter):
    """Apply paper_norm_trimmed_mean(alpha, beta) while caching a frozen native attacker."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._cached_attack = None
        self._last_post_metrics: dict[str, float] | None = None
        self._round_idx = self._initial_round_idx()

    def reset(self, seed: int | None = None):
        self._cached_attack = None
        self._last_post_metrics = None
        initial = super().reset(seed=seed)
        self._round_idx = self._initial_round_idx()
        return initial

    def run_round(self, attack, defense, attack_decision, defense_decision, evaluate: bool = True):
        del defense, attack_decision
        self._round_idx += 1
        self._apply_defense_decision(defense_decision)
        if self._cached_attack is None:
            self._cached_attack = self._build_attack(attack, _attack_name(attack), None)
        should_evaluate = evaluate or self._last_summary is None
        summary = self.runner.run_round(
            self._round_idx,
            attack=self._cached_attack,
            evaluate=should_evaluate,
            attacker_action=None,
            defense_decision=defense_decision,
        )
        translated = self._translate_summary(summary)
        for key, value in self._post_metrics_from_cache_or_eval(defense_decision, should_evaluate=should_evaluate).items():
            setattr(translated, key, value)
        if should_evaluate:
            self._last_summary = translated
        return translated

    def _apply_defense_decision(self, defense_decision) -> None:
        self.runner.defender.defense_type = "paper_norm_trimmed_mean"
        self.runner.defender.clipped_median_norm = float(defense_decision.norm_bound_alpha)
        self.runner.defender.trimmed_mean_ratio = float(defense_decision.trimmed_mean_beta)

    def _evaluate_post_training_copy(self, defense_decision) -> dict[str, float]:
        epsilon = float(defense_decision.neuroclip_epsilon or 0.0)
        defended_model = apply_post_defense(self.runner.model, "neuroclip", epsilon)
        metrics = self.evaluate_model(defended_model, self.current_weights)
        return {
            "post_clean_loss": float(metrics["clean_loss"]),
            "post_clean_acc": float(metrics["clean_acc"]),
            "post_backdoor_acc": float(metrics.get("backdoor_acc", 0.0)),
        }

    def _post_metrics_from_cache_or_eval(self, defense_decision, *, should_evaluate: bool) -> dict[str, float]:
        if not should_evaluate and self._last_post_metrics is not None:
            return dict(self._last_post_metrics)
        metrics = self._evaluate_post_training_copy(defense_decision)
        self._last_post_metrics = dict(metrics)
        return metrics

    def _initial_round_idx(self) -> int:
        return int(getattr(self.config.runtime, "start_round_idx", 1) or 1) - 1
