from pathlib import Path

import pytest

from meta_sg.stackelberg.metrics import compare_learned_to_fixed_paper3d, format_progress_line, summarize_rows


def test_summarize_rows_reports_paper3d_metrics(tmp_path: Path):
    rows = [
        {
            "defender_reward": -4.0,
            "post_clean_loss": 4.0,
            "post_clean_acc": 0.2,
            "alpha": 4.0,
            "beta": 0.2,
            "epsilon": 5.0,
        },
        {
            "defender_reward": -3.0,
            "post_clean_loss": 3.0,
            "post_clean_acc": 0.3,
            "alpha": 5.0,
            "beta": 0.1,
            "epsilon": 4.0,
        },
    ]

    summary = summarize_rows(rows, tmp_path)

    assert summary["mean_defender_reward"] == pytest.approx(-3.5)
    assert summary["mean_post_clean_loss"] == pytest.approx(3.5)
    assert summary["final_post_clean_acc"] == pytest.approx(0.3)
    assert summary["mean_post_clean_acc"] == pytest.approx(0.25)
    assert summary["mean_alpha"] == pytest.approx(4.5)
    assert summary["mean_beta"] == pytest.approx(0.15)
    assert summary["mean_epsilon"] == pytest.approx(4.5)


def test_compare_learned_to_fixed_paper3d_uses_reward_then_loss():
    comparison = compare_learned_to_fixed_paper3d(
        learned={"mean_defender_reward": -3.0, "mean_post_clean_loss": 3.0},
        baselines={
            "weak": {"mean_defender_reward": -5.0, "mean_post_clean_loss": 5.0},
            "strong": {"mean_defender_reward": -3.2, "mean_post_clean_loss": 3.2},
        },
    )

    assert comparison["best_fixed_baseline"] == "strong"
    assert comparison["defender_reward_improvement_vs_best_fixed"] == pytest.approx(0.2)
    assert comparison["post_clean_loss_delta_vs_best_fixed"] == pytest.approx(-0.2)


def test_format_progress_line_reports_paper3d_state():
    row = {
        "step": 7,
        "round": 107,
        "defender_reward": -2.5,
        "post_clean_loss": 2.5,
        "post_clean_acc": 0.25,
        "alpha": 4.0,
        "beta": 0.2,
        "epsilon": 5.0,
        "mean_malicious_norm": 3.5,
    }

    line = format_progress_line("train", row)

    assert "train" in line
    assert "step=7" in line
    assert "round=107" in line
    assert "reward=-2.500000" in line
    assert "post_loss=2.500000" in line
    assert "post_acc=0.250000" in line
    assert "alpha=4.0000" in line
    assert "beta=0.2000" in line
    assert "epsilon=5.0000" in line
    assert "mal_norm=3.5000" in line
