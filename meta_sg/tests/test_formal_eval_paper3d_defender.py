from types import SimpleNamespace

import numpy as np

from meta_sg.scripts.formal_eval_paper3d_defender import (
    compare_formal,
    parse_one_fixed_trim,
    reset_eval_episode_from_warmup,
)


def test_parse_one_fixed_trim_returns_named_paper3d_action():
    name, action = parse_one_fixed_trim("fixed_trim=4,0.2,5")

    assert name == "fixed_trim"
    assert action.alpha == 4.0
    assert action.beta == 0.2
    assert action.epsilon == 5.0


def test_compare_formal_reports_learned_and_fixed_deltas():
    summary = {
        "frozen_attacker_no_defense": {
            "mean_post_clean_acc": 0.10,
            "mean_post_clean_loss": 9.8,
        },
        "frozen_attacker_fixed_trim": {
            "mean_post_clean_acc": 0.12,
            "mean_post_clean_loss": 9.2,
        },
        "frozen_attacker_learned_defender": {
            "mean_post_clean_acc": 0.14,
            "mean_post_clean_loss": 9.1,
        },
    }

    comparison = compare_formal(summary)

    assert comparison["learned_acc_delta_vs_fixed_trim"] == 0.020000000000000018
    assert comparison["learned_loss_delta_vs_fixed_trim"] == -0.09999999999999964
    assert comparison["fixed_trim_acc_delta_vs_no_defense"] == 0.01999999999999999
    assert comparison["fixed_trim_loss_delta_vs_no_defense"] == -0.6000000000000014


def test_reset_eval_episode_from_warmup_restores_weights_and_round_index():
    class FakeCoordinator:
        def __init__(self):
            self.reset_seed = None
            self.current_weights = [np.array([0.0])]
            self.config = SimpleNamespace(runtime=SimpleNamespace(start_round_idx=101))
            self._round_idx = 0

        def reset(self, seed=None):
            self.reset_seed = seed
            self.current_weights = [np.array([-1.0])]
            return SimpleNamespace(weights=self.current_weights)

        def restore(self, snapshot):
            self._round_idx = snapshot.round_idx
            self.current_weights = [w.copy() for w in snapshot.weights]

    env = SimpleNamespace(coordinator=FakeCoordinator())
    warmup_weights = [np.array([42.0])]

    obs = reset_eval_episode_from_warmup(env, seed=123, warmup_weights=warmup_weights)

    assert env.coordinator.reset_seed == 123
    assert env.coordinator._round_idx == 100
    assert env.coordinator.current_weights[0][0] == 42.0
    assert obs[0][0] == 42.0
