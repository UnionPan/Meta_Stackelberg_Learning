from types import SimpleNamespace

import numpy as np

from meta_sg.stackelberg.warmup import clone_weights, reset_env_from_weights


def test_clone_weights_returns_independent_numpy_copies():
    original = [np.array([1.0])]

    cloned = clone_weights(original)
    cloned[0][0] = 2.0

    assert original[0][0] == 1.0


def test_reset_env_from_weights_restores_weights_round_and_observation():
    class FakeCoordinator:
        def __init__(self):
            self.config = SimpleNamespace(runtime=SimpleNamespace(start_round_idx=101))
            self.current_weights = [np.array([0.0])]
            self._round_idx = 0

        def reset(self, seed=None):
            self.seed = seed
            self.current_weights = [np.array([-1.0])]

        def restore(self, snapshot):
            self._round_idx = snapshot.round_idx
            self.current_weights = clone_weights(snapshot.weights)

    class FakeEnv:
        def __init__(self):
            self.coordinator = FakeCoordinator()
            self._round = 999
            self._history = [np.array([1.0])]

        def reset(self, seed=None):
            self.coordinator.reset(seed=seed)

        def _make_obs(self, weights):
            return np.asarray([weights[0][0] + 1.0], dtype=np.float32)

    env = FakeEnv()
    weights = [np.array([41.0])]

    obs = reset_env_from_weights(env, seed=123, weights=weights)

    assert env.coordinator.seed == 123
    assert env.coordinator._round_idx == 100
    assert env.coordinator.current_weights[0][0] == 41.0
    assert env._round == 0
    assert env._history == []
    assert obs.tolist() == [42.0]
