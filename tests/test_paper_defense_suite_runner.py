from pathlib import Path

from fl_sandbox.scripts.run_paper_defense_suite import (
    ALL_DEFENSES,
    DefensePlan,
    build_clean_command,
    build_fixed_eval_command,
    build_train_command,
    defense_plan_for,
    run_name_for,
    numeric_series_from_payload,
)


def test_run_name_uses_attack_defense_split_and_rounds():
    assert run_name_for("rl", "krum", rounds=500) == "mnist_rl_krum_paper_q_q0.1_500r"


def test_non_clipped_defense_train_uses_canonical_semantics_and_checkpoints():
    plan = DefensePlan(name="krum", semantics="canonical")
    cmd = build_train_command(
        plan,
        config_path=Path("cfg.yaml"),
        output_root=Path("out/train"),
        tb_root=Path("runs/train"),
        rounds=500,
        distribution_steps=50,
        attack_start_round=51,
        policy_train_end_round=200,
        policy_train_steps_per_round=10,
        simulator_horizon=3,
        checkpoint_interval=25,
    )

    assert "--defense_type" in cmd
    assert cmd[cmd.index("--defense_type") + 1] == "krum"
    assert cmd[cmd.index("--rl_attacker_semantics") + 1] == "canonical"
    assert cmd[cmd.index("--rl_checkpoint_interval") + 1] == "25"
    assert "--no-rl_freeze_policy" not in cmd


def test_all_defenses_includes_every_supported_aggregator():
    assert ALL_DEFENSES == (
        "fedavg",
        "krum",
        "multi_krum",
        "median",
        "clipped_median",
        "geometric_median",
        "trimmed_mean",
        "fltrust",
        "paper_norm_trimmed_mean",
    )


def test_paper_defense_plans_use_strict_reproduction_semantics_where_available():
    assert defense_plan_for("clipped_median").semantics == "legacy_clipped_median_strict"
    assert defense_plan_for("krum").semantics == "legacy_krum_strict"
    assert defense_plan_for("median").semantics == "canonical"


def test_fixed_eval_freezes_loaded_policy_and_disables_training_window():
    plan = DefensePlan(name="fltrust", semantics="canonical")
    checkpoint = Path("out/train/mnist_rl_fltrust_paper_q_q0.1_500r/checkpoints/rl_policy_latest.pt")
    cmd = build_fixed_eval_command(
        plan,
        config_path=Path("cfg.yaml"),
        output_root=Path("out/eval"),
        tb_root=Path("runs/eval"),
        policy_checkpoint=checkpoint,
        rounds=500,
        distribution_steps=50,
        attack_start_round=51,
        policy_train_steps_per_round=10,
        simulator_horizon=3,
        checkpoint_interval=100,
    )

    assert cmd[cmd.index("--rl_policy_train_end_round") + 1] == "0"
    assert cmd[cmd.index("--rl_policy_checkpoint_path") + 1] == str(checkpoint)
    assert "--rl_freeze_policy" in cmd
    assert cmd[cmd.index("--defense_type") + 1] == "fltrust"


def test_clean_command_keeps_same_defense_and_scale():
    plan = DefensePlan(name="median", semantics="canonical")
    cmd = build_clean_command(
        plan,
        config_path=Path("cfg.yaml"),
        output_root=Path("out/clean"),
        tb_root=Path("runs/clean"),
        rounds=500,
        distribution_steps=50,
        attack_start_round=51,
        policy_train_end_round=200,
    )

    assert cmd[cmd.index("--attack_type") + 1] == "clean"
    assert cmd[cmd.index("--defense_type") + 1] == "median"
    assert cmd[cmd.index("--rounds") + 1] == "500"


def test_numeric_series_ignores_text_metadata_series():
    payload = {
        "series": {
            "clean_acc": [0.1, 0.2],
            "defense_name": ["krum", "krum"],
        }
    }

    assert numeric_series_from_payload(payload) == {"clean_acc": [0.1, 0.2]}
