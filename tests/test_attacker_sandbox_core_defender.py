import unittest

from fl_sandbox.config.schema import DefenderSection
from fl_sandbox.core.defender import (
    DEFENSE_CHOICES,
    SandboxDefender,
    TrimmedMeanDefender,
    build_defender_config_kwargs,
    create_defender,
)


class TestAttackerSandboxCoreDefender(unittest.TestCase):

    def test_exposes_supported_defense_choices(self):
        self.assertEqual(
            DEFENSE_CHOICES,
            (
                "fedavg",
                "krum",
                "multi_krum",
                "median",
                "clipped_median",
                "geometric_median",
                "trimmed_mean",
                "fltrust",
                "paper_norm_trimmed_mean",
            ),
        )

    def test_build_defender_config_kwargs_maps_schema_fields(self):
        defender = DefenderSection(
            type="trimmed_mean",
            krum_attackers=3,
            multi_krum_selected=5,
            clipped_median_norm=1.5,
            trimmed_mean_ratio=0.3,
            geometric_median_iters=7,
            fltrust_root_size=128,
        )

        self.assertEqual(
            build_defender_config_kwargs(defender),
            {
                "defense_type": "trimmed_mean",
                "krum_attackers": 3,
                "multi_krum_selected": 5,
                "clipped_median_norm": 1.5,
                "trimmed_mean_ratio": 0.3,
                "geometric_median_iters": 7,
                "fltrust_root_size": 128,
            },
        )

    def test_create_defender_returns_typed_subclass(self):
        defender = create_defender(DefenderSection(type="trimmed_mean", trimmed_mean_ratio=0.3))

        self.assertIsInstance(defender, SandboxDefender)
        self.assertIsInstance(defender, TrimmedMeanDefender)
        self.assertEqual(defender.build_config_kwargs(), {"defense_type": "trimmed_mean", "trimmed_mean_ratio": 0.3})


if __name__ == "__main__":
    unittest.main()
