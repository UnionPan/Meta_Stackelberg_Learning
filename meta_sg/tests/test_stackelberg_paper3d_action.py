import argparse

import numpy as np
import pytest

from meta_sg.stackelberg.paper3d_action import (
    Paper3DAction,
    fixed_paper3d_to_raw_action,
    parse_fixed_paper3d_baselines,
    raw_action_to_paper3d,
)


def _args():
    return argparse.Namespace(
        alpha_min=0.1,
        alpha_max=30.0,
        beta_min=0.0,
        beta_max=0.45,
        neuroclip_eps_min=0.1,
        neuroclip_eps_max=10.0,
    )


def test_raw_action_decodes_to_paper3d_ranges():
    low = raw_action_to_paper3d(np.asarray([-1.0, -1.0, -1.0], dtype=np.float32), _args())
    mid = raw_action_to_paper3d(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), _args())
    high = raw_action_to_paper3d(np.asarray([1.0, 1.0, 1.0], dtype=np.float32), _args())

    assert low.alpha == pytest.approx(0.1)
    assert low.beta == pytest.approx(0.0)
    assert low.epsilon == pytest.approx(0.1)
    assert mid.alpha == pytest.approx(15.05)
    assert mid.beta == pytest.approx(0.225)
    assert mid.epsilon == pytest.approx(5.05)
    assert high.alpha == pytest.approx(30.0)
    assert high.beta == pytest.approx(0.45)
    assert high.epsilon == pytest.approx(10.0)


def test_fixed_paper3d_action_round_trips_to_raw():
    action = Paper3DAction(alpha=4.0, beta=0.2, epsilon=5.0)
    raw = fixed_paper3d_to_raw_action(action, _args())
    decoded = raw_action_to_paper3d(raw, _args())

    assert raw.shape == (3,)
    assert decoded.alpha == pytest.approx(4.0)
    assert decoded.beta == pytest.approx(0.2)
    assert decoded.epsilon == pytest.approx(5.0)


def test_parse_fixed_paper3d_baselines():
    baselines = parse_fixed_paper3d_baselines(["norm_trim=4,0.2,10", "neuro=4,0.2,2"])

    assert baselines["norm_trim"] == Paper3DAction(alpha=4.0, beta=0.2, epsilon=10.0)
    assert baselines["neuro"] == Paper3DAction(alpha=4.0, beta=0.2, epsilon=2.0)
