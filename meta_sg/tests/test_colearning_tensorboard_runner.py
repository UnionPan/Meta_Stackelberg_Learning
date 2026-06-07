import pytest

from meta_sg.scripts.run_m1_m2_tensorboard import total_training_rounds
from meta_sg.scripts.run_meta_sg_colearning_tensorboard import (
    total_colearning_training_rounds,
)
from meta_sg.scripts.run_defender_comparison_tensorboard import (
    DEFENDER_SPECS,
    decode_defender_raw_action,
    paper_aligned_defender_config,
    safe_defender_config,
    training_stage_round_offsets,
    total_defender_comparison_training_rounds,
)


def test_total_training_rounds_counts_all_radius_iterations_and_episodes():
    assert total_training_rounds(
        num_radii=5,
        outer_iters=10,
        horizon=4,
        br_episodes=5,
    ) == 1000


def test_total_colearning_training_rounds_counts_pre_and_post_br_rollouts():
    assert total_colearning_training_rounds(
        outer_iters=20,
        tasks_per_iter=2,
        horizon=5,
        post_br_defender_updates=1,
    ) == 400


def test_defender_comparison_uses_four_3d_defender_specs():
    assert [spec.name for spec in DEFENDER_SPECS] == [
        "fixed_mid_defender",
        "fixed_strong_clip_defender",
        "learned_meta_defender",
        "colearning_defender",
    ]
    for spec in DEFENDER_SPECS:
        assert spec.raw_action.shape == (3,)

    mid = decode_defender_raw_action(DEFENDER_SPECS[0].raw_action)
    strong = decode_defender_raw_action(DEFENDER_SPECS[1].raw_action)

    assert mid.norm_bound_alpha == pytest.approx(2.5)
    assert mid.trimmed_mean_beta == pytest.approx(0.225)
    assert strong.norm_bound_alpha < mid.norm_bound_alpha
    assert strong.trimmed_mean_beta > mid.trimmed_mean_beta


def test_safe_defender_config_keeps_alpha_away_from_zero():
    cfg = safe_defender_config()

    low = decode_defender_raw_action([-1.0, -1.0, -1.0], config=cfg)
    mid = decode_defender_raw_action([0.0, 0.0, 0.0], config=cfg)

    assert low.norm_bound_alpha == pytest.approx(0.5)
    assert low.trimmed_mean_beta == pytest.approx(0.0)
    assert mid.norm_bound_alpha == pytest.approx(2.75)
    assert mid.trimmed_mean_beta == pytest.approx(0.2)


def test_paper_aligned_defender_config_uses_relative_alpha_and_loss_reward():
    cfg = paper_aligned_defender_config()

    assert cfg.relative_alpha
    assert cfg.reward_mode == "loss"
    assert cfg.alpha_min == pytest.approx(1e-6)


def test_total_defender_comparison_training_rounds_counts_shared_training_and_eval():
    assert total_defender_comparison_training_rounds(
        shared_attacker_episodes=2,
        learned_meta_iters=3,
        colearning_iters=4,
        defender_count=4,
        seeds=2,
        train_horizon=5,
        eval_horizon=7,
        colearning_post_br_defender_updates=1,
    ) == 2 * 5 + 3 * 5 + 4 * 5 * 2 + 4 * 2 * 7


def test_training_stage_round_offsets_use_actual_cumulative_fl_rounds():
    assert training_stage_round_offsets(
        shared_attacker_episodes=2,
        learned_meta_iters=3,
        train_horizon=5,
        colearning_post_br_defender_updates=1,
    ) == {
        "shared_attacker": 0,
        "learned_meta_defender": 10,
        "colearning_defender": 25,
    }
