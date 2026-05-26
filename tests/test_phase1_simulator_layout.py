from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning import core as distribution_learning
from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning import generate_distribution


def test_phase1_distribution_learning_lives_under_simulator_package():
    assert distribution_learning.PaperGradientReconstructor.__module__.endswith(
        "simulator.distribution_learning.core"
    )
    assert distribution_learning.write_distribution_artifacts.__module__.endswith(
        "simulator.distribution_learning.core"
    )


def test_generate_distribution_entrypoints_live_under_simulator_package():
    assert generate_distribution.parse_args.__module__.endswith("simulator.distribution_learning.generate_distribution")
    assert generate_distribution.generate_distribution.__module__.endswith("simulator.distribution_learning.generate_distribution")
    assert generate_distribution.sample_pre_initial_batch.__module__.endswith("simulator.distribution_learning.generate_distribution")
