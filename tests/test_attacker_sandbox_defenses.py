import unittest

import numpy as np

from fl_sandbox.defenders import AggregationDefender, fltrust_aggregate, median_aggregate


def _weights(value: float):
    return [np.asarray([value], dtype=np.float32)]


class TestAttackerSandboxDefenses(unittest.TestCase):

    def test_median_rejects_single_outlier(self):
        old = _weights(0.0)
        updates = [_weights(1.0), _weights(1.2), _weights(50.0)]

        aggregated = median_aggregate(old, updates)

        self.assertTrue(np.allclose(aggregated[0], np.asarray([1.2], dtype=np.float32)))

    def test_krum_picks_benign_cluster(self):
        old = _weights(0.0)
        defender = AggregationDefender(defense_type="krum", krum_attackers=1)
        updates = [_weights(1.0), _weights(1.1), _weights(30.0), _weights(0.9)]

        aggregated = defender.aggregate(old, updates)

        self.assertLess(abs(float(aggregated[0][0]) - 1.0), 0.25)

    def test_clipped_median_limits_large_norm(self):
        old = _weights(0.0)
        defender = AggregationDefender(defense_type="clipped_median", clipped_median_norm=2.0)
        updates = [_weights(1.0), _weights(1.5), _weights(100.0)]

        aggregated = defender.aggregate(old, updates)

        self.assertLessEqual(abs(float(aggregated[0][0])), 2.0)

    def test_fltrust_falls_back_to_fedavg_without_reference(self):
        old = _weights(0.0)
        updates = [_weights(1.0), _weights(3.0)]

        aggregated = fltrust_aggregate(old, updates, trusted_weights=None)

        self.assertTrue(np.allclose(aggregated[0], np.asarray([2.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
