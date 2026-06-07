from types import SimpleNamespace

import numpy as np
import pytest

from meta_sg.stackelberg.paper3d_sandbox import Paper3DSandboxCoordinator
from meta_sg.simulation.fl_sandbox_adapter import SandboxConfig
from meta_sg.strategies.types import DefenseDecision


def test_paper3d_coordinator_starts_at_attack_window():
    coordinator = Paper3DSandboxCoordinator(SandboxConfig(start_round_idx=101))

    assert coordinator._initial_round_idx() == 100


def test_paper3d_coordinator_maps_defense_decision_to_sandbox_fields():
    coordinator = object.__new__(Paper3DSandboxCoordinator)
    coordinator.runner = SimpleNamespace(
        defender=SimpleNamespace(
            defense_type="clipped_median",
            clipped_median_norm=2.0,
            trimmed_mean_ratio=0.0,
        )
    )

    coordinator._apply_defense_decision(
        DefenseDecision(norm_bound_alpha=4.0, trimmed_mean_beta=0.2, neuroclip_epsilon=5.0)
    )

    assert coordinator.runner.defender.defense_type == "paper_norm_trimmed_mean"
    assert coordinator.runner.defender.clipped_median_norm == pytest.approx(4.0)
    assert coordinator.runner.defender.trimmed_mean_ratio == pytest.approx(0.2)


def test_paper3d_post_eval_reuses_cached_metrics_when_not_evaluating():
    coordinator = object.__new__(Paper3DSandboxCoordinator)
    coordinator._last_post_metrics = {
        "post_clean_loss": 1.5,
        "post_clean_acc": 0.6,
        "post_backdoor_acc": 0.2,
    }
    coordinator.evaluate_weights = lambda weights: (_ for _ in ()).throw(AssertionError("should not evaluate"))

    metrics = coordinator._post_metrics_from_cache_or_eval(
        DefenseDecision(norm_bound_alpha=4.0, trimmed_mean_beta=0.2, neuroclip_epsilon=5.0),
        should_evaluate=False,
    )

    assert metrics == {
        "post_clean_loss": 1.5,
        "post_clean_acc": 0.6,
        "post_backdoor_acc": 0.2,
    }


def test_paper3d_post_eval_uses_model_aware_neuroclip(monkeypatch):
    coordinator = object.__new__(Paper3DSandboxCoordinator)
    weights = [np.asarray([1.0], dtype=np.float32)]
    coordinator.runner = SimpleNamespace(model=object(), current_weights=weights)
    captured = {}

    def fake_apply_post_defense(model, defense_type, param, eval_loader=None, device=None):
        captured["model"] = model
        captured["defense_type"] = defense_type
        captured["param"] = param
        return "defended-model"

    def fake_evaluate_model(model, weights):
        captured["evaluated_model"] = model
        captured["weights"] = weights
        return {
            "clean_loss": 2.5,
            "clean_acc": 0.75,
            "backdoor_acc": 0.1,
        }

    monkeypatch.setattr("meta_sg.stackelberg.paper3d_sandbox.apply_post_defense", fake_apply_post_defense)
    coordinator.evaluate_model = fake_evaluate_model

    metrics = coordinator._evaluate_post_training_copy(
        DefenseDecision(norm_bound_alpha=4.0, trimmed_mean_beta=0.2, neuroclip_epsilon=3.0)
    )

    assert captured == {
        "model": coordinator.runner.model,
        "defense_type": "neuroclip",
        "param": 3.0,
        "evaluated_model": "defended-model",
        "weights": weights,
    }
    assert metrics == {
        "post_clean_loss": 2.5,
        "post_clean_acc": 0.75,
        "post_backdoor_acc": 0.1,
    }
