"""Smoke tests for Layer 1: FL simulation stub."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from meta_sg.simulation.fl_sandbox_adapter import FLSandboxCoordinatorAdapter
from meta_sg.simulation.stub import StubCoordinator, DEFAULT_LAYER_SHAPES
from meta_sg.simulation.types import RoundSummary, SimulationSnapshot
from meta_sg.strategies.attacks.fixed import IPMAttack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackDecision, DefenseDecision, ATTACK_DOMAIN
from meta_sg.games.bsmg_env import BSMGConfig, BSMGEnv


def _make_coord(**kw):
    return StubCoordinator(num_clients=6, num_attackers=2, seed=0, **kw)


def _make_decisions():
    d = DefenseDecision.from_raw(np.array([0.0, 0.0, 0.5], dtype=np.float32))
    a = AttackDecision.from_raw(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return d, a


class PostMetricStubCoordinator(StubCoordinator):
    def run_round(self, *args, **kwargs):
        summary = super().run_round(*args, **kwargs)
        summary.post_clean_loss = 1.25
        summary.post_clean_acc = 0.75
        summary.post_backdoor_acc = 0.1
        return summary


# ── reset ─────────────────────────────────────────────────────────────────

def test_reset_returns_weights_for_every_layer():
    coord = _make_coord()
    init = coord.reset()
    assert len(init.weights) == len(DEFAULT_LAYER_SHAPES)


def test_reset_shapes_match_layer_shapes():
    coord = _make_coord()
    init = coord.reset()
    for w, shape in zip(init.weights, DEFAULT_LAYER_SHAPES):
        assert w.shape == shape, f"Expected {shape}, got {w.shape}"


def test_reset_round_idx_is_zero():
    coord = _make_coord()
    init = coord.reset()
    assert init.round_idx == 0


def test_reset_different_seed_different_weights():
    coord = _make_coord()
    w0 = coord.reset(seed=0).weights
    w1 = coord.reset(seed=1).weights
    assert not np.allclose(
        np.concatenate([w.ravel() for w in w0]),
        np.concatenate([w.ravel() for w in w1]),
    )


# ── run_round ─────────────────────────────────────────────────────────────

def test_run_round_returns_round_summary():
    coord = _make_coord()
    coord.reset(seed=0)
    d, a = _make_decisions()
    summary = coord.run_round(
        attack=IPMAttack(),
        defense=PaperDefenseStrategy(),
        attack_decision=a,
        defense_decision=d,
    )
    assert isinstance(summary, RoundSummary)


def test_fl_sandbox_adapter_translate_summary_preserves_attack_metrics():
    adapter = FLSandboxCoordinatorAdapter.__new__(FLSandboxCoordinatorAdapter)
    adapter._last_summary = None
    summary = SimpleNamespace(
        round_idx=1,
        clean_acc=0.8,
        backdoor_acc=0.0,
        clean_loss=1.2,
        attack_name="rl",
        defense_name="paper_norm_trimmed_mean",
        benign_update_norms=[],
        malicious_update_norms=[],
        malicious_cosines_to_benign=[],
        malicious_cosines_to_aggregate=[],
        selected_attackers=[],
        sampled_clients=[],
        attack_metrics={
            "rl_action_gamma": 11.0,
            "rl_action_local_steps": 19.0,
        },
    )

    translated = adapter._translate_summary(summary)

    assert translated.attack_metrics["rl_action_gamma"] == pytest.approx(11.0)
    assert translated.attack_metrics["rl_action_local_steps"] == pytest.approx(19.0)


def test_run_round_increments_round_idx():
    coord = _make_coord()
    coord.reset()
    d, a = _make_decisions()
    kwargs = dict(attack=IPMAttack(), defense=PaperDefenseStrategy(),
                  attack_decision=a, defense_decision=d)
    coord.run_round(**kwargs)
    coord.run_round(**kwargs)
    summary = coord.run_round(**kwargs)
    assert summary.round_idx == 3


def test_run_round_metrics_in_valid_range():
    coord = _make_coord()
    coord.reset()
    d, a = _make_decisions()
    summary = coord.run_round(
        attack=IPMAttack(), defense=PaperDefenseStrategy(),
        attack_decision=a, defense_decision=d,
    )
    assert 0.0 <= summary.clean_acc <= 1.0
    assert 0.0 <= summary.backdoor_acc <= 1.0


def test_run_round_changes_weights():
    coord = _make_coord()
    coord.reset()
    before = np.concatenate([w.ravel() for w in coord.current_weights])
    d, a = _make_decisions()
    coord.run_round(
        attack=IPMAttack(), defense=PaperDefenseStrategy(),
        attack_decision=a, defense_decision=d,
    )
    after = np.concatenate([w.ravel() for w in coord.current_weights])
    assert not np.allclose(before, after), "Weights must change after a round"


# ── current_weights returns copy ──────────────────────────────────────────

def test_current_weights_is_copy():
    coord = _make_coord()
    coord.reset()
    w1 = coord.current_weights
    w1[0][:] = 999.0
    w2 = coord.current_weights
    assert not np.allclose(w2[0], 999.0), "current_weights must return a copy"


# ── snapshot / restore ────────────────────────────────────────────────────

def test_snapshot_restore_roundtrip():
    coord = _make_coord()
    coord.reset(seed=7)
    d, a = _make_decisions()
    snap = coord.snapshot()
    coord.run_round(attack=IPMAttack(), defense=PaperDefenseStrategy(),
                    attack_decision=a, defense_decision=d)
    coord.restore(snap)
    restored_vec = np.concatenate([w.ravel() for w in coord.current_weights])
    snap_vec = np.concatenate([w.ravel() for w in snap.weights])
    assert np.allclose(restored_vec, snap_vec)


def test_snapshot_restore_round_idx():
    coord = _make_coord()
    coord.reset()
    d, a = _make_decisions()
    coord.run_round(attack=IPMAttack(), defense=PaperDefenseStrategy(),
                    attack_decision=a, defense_decision=d)
    snap = coord.snapshot()
    assert snap.round_idx == 1
    coord.run_round(attack=IPMAttack(), defense=PaperDefenseStrategy(),
                    attack_decision=a, defense_decision=d)
    coord.restore(snap)
    assert coord.snapshot().round_idx == 1


def test_snapshot_is_independent_copy():
    coord = _make_coord()
    coord.reset()
    snap = coord.snapshot()
    # Mutating snap.weights should not affect coordinator
    snap.weights[0][:] = 888.0
    w = coord.current_weights[0]
    assert not np.allclose(w, 888.0)


def test_bsmg_env_can_penalize_extreme_defender_actions():
    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1, action_prior_weight=0.5),
    )
    env.reset(seed=3)

    _, reward, _, _, info = env.step(
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert info["action_prior_penalty"] == pytest.approx(0.5)
    assert reward == pytest.approx(info["clean_acc"] - info["backdoor_acc"] - 0.5)


def test_bsmg_env_can_penalize_low_server_lr():
    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=1,
            third_action="both",
            server_lr_min=0.6,
            server_lr_max=1.0,
            server_lr_penalty_weight=0.5,
        ),
    )
    env.reset(seed=3)

    _, reward, _, _, info = env.step(
        np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert info["defense_decision"].server_lr == pytest.approx(0.6)
    assert info["server_lr_penalty"] == pytest.approx(0.5 * (1.0 - 0.6) ** 2)
    assert reward == pytest.approx(
        info["clean_acc"] - info["backdoor_acc"] - info["server_lr_penalty"]
    )


def test_defense_decision_can_decode_neuroclip_and_server_lr_from_four_dim_action():
    decision = DefenseDecision.from_raw(
        np.asarray([0.0, 0.0, -1.0, 1.0], dtype=np.float32),
        third_action="both",
        eps_min=2.0,
        eps_max=10.0,
        server_lr_min=0.1,
        server_lr_max=0.5,
    )

    assert decision.norm_bound_alpha == pytest.approx(2.5)
    assert decision.trimmed_mean_beta == pytest.approx(0.225)
    assert decision.neuroclip_epsilon == pytest.approx(2.0)
    assert decision.server_lr == pytest.approx(0.5)


def test_bsmg_env_uses_four_defender_actions_for_combined_defense():
    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1, third_action="both"),
    )

    assert env.act_dim == 4
    assert env.attacker_act_dim == 3


def test_bsmg_env_can_use_loss_based_defender_reward():
    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1, reward_mode="loss"),
    )
    env.reset(seed=4)

    _, reward, _, _, info = env.step(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert reward == pytest.approx(-info["clean_loss"] - info["backdoor_acc"])


def test_bsmg_env_paper_aligned_state_action_reward():
    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["rl"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(
            horizon=1,
            num_tail_layers=2,
            normalise_obs=True,
            history_len=0,
            reward_mode="loss",
            lambda_bd=0.0,
            action_prior_weight=0.0,
        ),
    )

    obs = env.reset(seed=123)
    next_obs, reward, _, _, info = env.step(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert env.act_dim == 3
    assert obs.ndim == 1
    assert next_obs.shape == obs.shape
    assert reward == pytest.approx(-info["clean_loss"])
    assert "defense_decision" in info


def test_bsmg_env_uses_post_training_evaluator_for_reward_and_info():
    calls = []

    def evaluator(weights):
        calls.append(weights)
        return {
            "clean_acc": 0.9,
            "clean_loss": 0.2,
            "backdoor_acc": 0.4,
        }

    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1, lambda_bd=0.0, reward_mode="accuracy"),
        evaluator=evaluator,
    )
    env.reset(seed=7)

    _, reward, attacker_reward, _, info = env.step(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert len(calls) == 1
    assert reward == pytest.approx(0.9)
    assert attacker_reward == pytest.approx(-0.9)
    assert info["clean_acc"] == pytest.approx(0.9)
    assert info["clean_loss"] == pytest.approx(0.2)
    assert info["backdoor_acc"] == pytest.approx(0.4)
    assert info["post_clean_acc"] == pytest.approx(0.9)
    assert info["post_clean_loss"] == pytest.approx(0.2)
    assert info["post_backdoor_acc"] == pytest.approx(0.4)
    assert "pre_clean_acc" in info


def test_bsmg_env_raises_when_post_training_evaluator_fails():
    def evaluator(_weights):
        raise RuntimeError("eval failed")

    env = BSMGEnv(
        coordinator=_make_coord(),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1),
        evaluator=evaluator,
    )
    env.reset(seed=8)

    with pytest.raises(RuntimeError, match="eval failed"):
        env.step(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))


def test_stub_evaluate_weights_is_side_effect_free():
    coord = _make_coord()
    coord.reset(seed=0)
    before = coord._base_clean_acc

    first = coord.evaluate_weights(coord.current_weights)
    second = coord.evaluate_weights(coord.current_weights)

    assert coord._base_clean_acc == pytest.approx(before)
    assert first == second
    assert set(first) == {"clean_acc", "clean_loss", "backdoor_acc"}


def test_bsmg_env_includes_post_training_metrics_when_available():
    env = BSMGEnv(
        coordinator=PostMetricStubCoordinator(num_clients=6, num_attackers=2, seed=0),
        attack_type=ATTACK_DOMAIN["ipm"],
        attack_strategy=IPMAttack(),
        defense_strategy=PaperDefenseStrategy(),
        config=BSMGConfig(horizon=1),
    )
    env.reset(seed=5)

    _, _, _, _, info = env.step(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    assert info["post_clean_loss"] == pytest.approx(1.25)
    assert info["post_clean_acc"] == pytest.approx(0.75)
    assert info["post_backdoor_acc"] == pytest.approx(0.1)
