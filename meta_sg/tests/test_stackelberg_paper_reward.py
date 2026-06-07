import numpy as np
import pytest

from meta_sg.stackelberg.paper_reward import compute_paper_defender_reward
from meta_sg.strategies.defenses.paper import apply_neuroclip


def test_apply_neuroclip_clips_weight_copy_without_mutating_input():
    weights = [np.asarray([-2.0, -0.5, 0.5, 3.0], dtype=np.float32)]

    clipped = apply_neuroclip(weights, epsilon=1.0)

    assert clipped[0].tolist() == pytest.approx([-1.0, -0.5, 0.5, 1.0])
    assert weights[0].tolist() == pytest.approx([-2.0, -0.5, 0.5, 3.0])


def test_paper_defender_reward_is_negative_post_clean_loss_only():
    info = {
        "post_clean_loss": 4.25,
        "clean_acc": 0.8,
        "malicious_update_norms": [100.0],
    }

    assert compute_paper_defender_reward(info) == pytest.approx(-4.25)


def test_paper_defender_reward_supports_positive_scale_without_changing_sign():
    info = {"post_clean_loss": 4.25}

    assert compute_paper_defender_reward(info, scale=0.01) == pytest.approx(-0.0425)


def test_paper_reward_requires_post_clean_loss():
    with pytest.raises(KeyError):
        compute_paper_defender_reward({"clean_loss": 3.0})
