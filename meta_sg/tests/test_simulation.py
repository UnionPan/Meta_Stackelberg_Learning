"""Smoke tests for Layer 1: FL simulation stub."""
import copy
import numpy as np
import pytest

from meta_sg.simulation.stub import StubCoordinator, DEFAULT_LAYER_SHAPES
from meta_sg.simulation.types import RoundSummary, SimulationSnapshot
from meta_sg.strategies.attacks.fixed import IPMAttack
from meta_sg.strategies.defenses.paper import PaperDefenseStrategy
from meta_sg.strategies.types import AttackDecision, DefenseDecision, ATTACK_DOMAIN


def _make_coord(**kw):
    return StubCoordinator(num_clients=6, num_attackers=2, seed=0, **kw)


def _make_decisions():
    d = DefenseDecision.from_raw(np.array([0.0, 0.0, 0.5], dtype=np.float32))
    a = AttackDecision.from_raw(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return d, a


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
