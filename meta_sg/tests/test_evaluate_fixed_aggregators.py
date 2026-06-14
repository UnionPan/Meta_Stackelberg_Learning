from fl_sandbox.attacks.registry import create_attack
from meta_sg.scripts.evaluate_fixed_aggregators import Scenario, _attack_config, _filter_scenarios, _patched_config
from meta_sg.simulation.fl_sandbox_adapter import SandboxConfig


def test_attack_config_includes_default_attack_parameters_for_ipm_and_lmp():
    config = SandboxConfig(dataset="mnist", attack_type="clean", defense_type="fedavg")

    ipm_cfg = _attack_config(config, "ipm")
    lmp_cfg = _attack_config(config, "lmp")

    assert hasattr(ipm_cfg, "ipm_scaling")
    assert hasattr(lmp_cfg, "lmp_scale")
    assert create_attack(ipm_cfg).name.lower() == "ipm"
    assert create_attack(lmp_cfg).name.lower() == "lmp"


def test_attack_config_prefers_scenario_patch_values():
    config = SandboxConfig(dataset="mnist", attack_type="clean", defense_type="fedavg")
    patched = _patched_config(config, {"ipm_scaling": 4.0})

    attack_cfg = _attack_config(patched, "ipm")

    assert attack_cfg.ipm_scaling == 4.0


def test_filter_scenarios_keeps_requested_order_and_names():
    scenarios = [
        Scenario("clean", "clean", {}),
        Scenario("paper_ipm", "ipm", {}),
        Scenario("lmp", "lmp", {}),
    ]

    filtered = _filter_scenarios(scenarios, "lmp,clean")

    assert [scenario.name for scenario in filtered] == ["lmp", "clean"]
