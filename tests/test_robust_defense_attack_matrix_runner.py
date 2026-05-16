from pathlib import Path

from fl_sandbox.scripts.run_robust_defense_attack_matrix import (
    BENCHMARK_ATTACK_NAMES,
    DEFAULT_ATTACK_NAMES,
    OPTIMIZED_RL_ATTACK_NAMES,
    attack_plan_by_name,
    attack_plans_for_defense,
    build_attack_command,
    completed_row,
    parse_reuse_summary_specs,
    run_name_for,
    summary_payload_round_count,
    truncate_payload_rounds,
)


def test_matrix_filters_defense_specific_attack_plans():
    clipped = {plan.name for plan in attack_plans_for_defense("clipped_median")}
    krum = {plan.name for plan in attack_plans_for_defense("krum")}

    assert {"clean", "ipm", "lmp", "dba", "bfl"}.issubset(clipped)
    assert {"clean", "ipm", "lmp", "dba", "bfl"}.issubset(krum)
    assert "rl_clipped_median_scaleaware" in clipped
    assert "rl_krum_geometry" in krum
    assert "rl_clipped_median_strict" not in clipped
    assert "rl_krum_strict" not in krum
    assert "clipped_median_geometry_search" not in clipped
    assert "krum_geometry_search" not in krum
    assert "krum_geometry_search" not in clipped
    assert "clipped_median_geometry_search" not in krum
    assert "rl_krum_geometry" not in clipped
    assert "rl_clipped_median_scaleaware" not in krum


def test_matrix_default_attack_names_are_focused_benchmark_plus_two_optimized_rl_attackers():
    assert BENCHMARK_ATTACK_NAMES == ("clean", "ipm", "lmp", "dba", "bfl")
    assert OPTIMIZED_RL_ATTACK_NAMES == ("rl_clipped_median_scaleaware", "rl_krum_geometry")
    assert DEFAULT_ATTACK_NAMES == BENCHMARK_ATTACK_NAMES + OPTIMIZED_RL_ATTACK_NAMES


def test_matrix_can_request_strict_and_heuristic_plans_explicitly():
    clipped = {
        plan.name
        for plan in attack_plans_for_defense(
            "clipped_median",
            ["rl_clipped_median_strict", "clipped_median_geometry_search"],
        )
    }
    krum = {
        plan.name
        for plan in attack_plans_for_defense("krum", ["rl_krum_strict", "krum_geometry_search"])
    }

    assert clipped == {"clean", "rl_clipped_median_strict", "clipped_median_geometry_search"}
    assert krum == {"clean", "rl_krum_strict", "krum_geometry_search"}


def test_matrix_parses_reused_summary_specs():
    mapping = parse_reuse_summary_specs(
        [
            "clipped_median:rl_clipped_median_scaleaware:out/clipped/summary.json",
            "krum:rl_krum_geometry:out/krum",
        ]
    )

    assert mapping[("clipped_median", "rl_clipped_median_scaleaware")] == Path("out/clipped/summary.json")
    assert mapping[("krum", "rl_krum_geometry")] == Path("out/krum")


def test_matrix_truncates_reused_payload_series_to_requested_rounds():
    payload = {
        "series": {
            "clean_acc": [0.1, 0.2, 0.3, 0.4],
            "clean_loss": [2.0, 1.8, 1.6, 1.4],
            "label": ["a", "b", "c", "d"],
        },
        "rounds": [{"round": 1}, {"round": 2}, {"round": 3}],
    }

    limited = truncate_payload_rounds(payload, 2)

    assert limited["series"]["clean_acc"] == [0.1, 0.2]
    assert limited["series"]["clean_loss"] == [2.0, 1.8]
    assert limited["series"]["label"] == ["a", "b"]
    assert limited["rounds"] == [{"round": 1}, {"round": 2}]
    assert payload["series"]["clean_acc"] == [0.1, 0.2, 0.3, 0.4]


def test_matrix_counts_payload_rounds_from_numeric_series_or_round_rows():
    assert summary_payload_round_count({"series": {"clean_acc": [0.1, 0.2]}}) == 2
    assert summary_payload_round_count({"rounds": [{"round": 1}, {"round": 2}, {"round": 3}]}) == 3
    assert summary_payload_round_count({"series": {"name": ["a", "b"]}}) == 0


def test_matrix_run_name_keeps_plan_name_to_avoid_rl_collisions():
    assert (
        run_name_for("rl_krum_geometry", "krum", rounds=150)
        == "krum/rl_krum_geometry/mnist_rl_krum_paper_q_q0.1_150r"
    )


def test_matrix_builds_rl_geometry_command_with_paper_scale_knobs():
    plan = attack_plan_by_name("rl_krum_geometry")
    cmd = build_attack_command(
        plan,
        defense="krum",
        config_path=Path("cfg.yaml"),
        output_root=Path("out"),
        tb_root=Path("runs"),
        rounds=150,
        num_clients=100,
        num_attackers=20,
        subsample_rate=0.1,
        krum_attackers=20,
        distribution_steps=100,
        attack_start_round=101,
        policy_train_end_round=150,
        policy_train_steps_per_round=50,
        simulator_horizon=1000,
        checkpoint_interval=25,
        parallel_clients=1,
    )

    assert cmd[cmd.index("--attack_type") + 1] == "rl"
    assert cmd[cmd.index("--defense_type") + 1] == "krum"
    assert cmd[cmd.index("--rl_attacker_semantics") + 1] == "legacy_krum_geometry"
    assert cmd[cmd.index("--rl_policy_train_steps_per_round") + 1] == "50"
    assert cmd[cmd.index("--num_clients") + 1] == "100"
    assert cmd[cmd.index("--krum_attackers") + 1] == "20"
    assert "--rl_save_final_checkpoint" in cmd
    assert cmd[cmd.index("--output_root") + 1] == "out/krum/rl_krum_geometry"


def test_matrix_builds_heuristic_command_without_rl_semantics():
    plan = attack_plan_by_name("clipped_median_geometry_search")
    cmd = build_attack_command(
        plan,
        defense="clipped_median",
        config_path=Path("cfg.yaml"),
        output_root=Path("out"),
        tb_root=Path("runs"),
        rounds=40,
    )

    assert cmd[cmd.index("--attack_type") + 1] == "clipped_median_geometry_search"
    assert "--rl_attacker_semantics" not in cmd
    assert cmd[cmd.index("--output_root") + 1] == "out/clipped_median/clipped_median_geometry_search"


def test_matrix_completed_row_reports_clean_relative_drops_and_tail():
    clean_payload = {"series": {"clean_acc": [0.8, 0.9, 1.0], "clean_loss": [1.0, 0.5, 0.2]}}
    attack_payload = {
        "series": {
            "clean_acc": [0.7, 0.6, 0.5],
            "clean_loss": [1.2, 1.5, 1.8],
            "rl_loss/actor": [0.3, 0.2],
        }
    }

    row = completed_row(
        defense="krum",
        attack_name="rl_krum_geometry",
        benchmark="rl",
        clean_payload=clean_payload,
        attack_payload=attack_payload,
        tail_rounds=2,
    )

    assert row["defense"] == "krum"
    assert row["attack_name"] == "rl_krum_geometry"
    assert row["benchmark"] == "rl"
    assert row["clean_final_acc"] == 1.0
    assert row["attack_final_acc"] == 0.5
    assert row["final_acc_drop"] == 0.5
    assert row["clean_tail_acc"] == 0.95
    assert row["attack_tail_acc"] == 0.55
    assert row["tail_acc_drop"] == 0.4
