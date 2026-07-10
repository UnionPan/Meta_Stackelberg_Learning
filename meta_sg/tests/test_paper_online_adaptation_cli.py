import pytest
from pathlib import Path

from meta_sg.scripts.evaluate_meta_sg_direct import parse_args


def test_parse_accepts_clean_global_backdoor_mixed_scenario_set():
    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--scenario-set",
            "clean_global_backdoor_mixed",
        ]
    )

    assert args.scenario_set == "clean_global_backdoor_mixed"


def test_clean_global_backdoor_mixed_scenarios_match_training_domain():
    from meta_sg.scripts.evaluate_meta_sg_direct import _attack_context_names, _scenarios

    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--scenario-set",
            "clean_global_backdoor_mixed",
            "--attack-context",
        ]
    )

    assert [scenario.name for scenario in _scenarios(args)] == [
        "clean",
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
        "mixed_backdoor",
    ]
    assert _attack_context_names(args) == (
        "clean",
        "ipm",
        "lmp",
        "rl",
        "bfl",
        "dba",
        "rl_backdoor",
        "mixed_backdoor",
    )


def test_4d_global_backdoor_eval_job_runs_direct_and_proxy_adaptation():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_4d_both_global_backdoor_mixed_eval_job.sh"
    )
    text = script.read_text()

    assert "--scenario-set clean_global_backdoor_mixed" in text
    assert "--defender-third-action both" in text
    assert "--post-defense-mode model_aware_neuroclip" in text
    assert "--few-shot-method paper_online_proxy_td3" in text
    assert "direct_h200_clean_global_backdoor_mixed.json" in text
    assert "adapt_proxy_h200_clean_global_backdoor_mixed.json" in text


def test_backdoor_guarded_selection_requires_clean_safe_asr_reduction():
    from meta_sg.scripts.evaluate_meta_sg_direct import _backdoor_guarded_selection_decision

    decision = _backdoor_guarded_selection_decision(
        base_clean_acc=0.92,
        adapted_clean_acc=0.91,
        base_backdoor_acc=0.90,
        adapted_backdoor_acc=0.75,
        asr_reduction_margin=0.10,
        clean_drop_tolerance=0.02,
    )

    assert decision["accepted"] is True
    assert decision["selected"] == "adapted"
    assert decision["clean_safe"] is True
    assert decision["asr_reduction"] == 0.15


def test_backdoor_guarded_selection_rejects_when_asr_does_not_improve():
    from meta_sg.scripts.evaluate_meta_sg_direct import _backdoor_guarded_selection_decision

    decision = _backdoor_guarded_selection_decision(
        base_clean_acc=0.92,
        adapted_clean_acc=0.93,
        base_backdoor_acc=0.90,
        adapted_backdoor_acc=0.88,
        asr_reduction_margin=0.05,
        clean_drop_tolerance=0.02,
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "base"
    assert decision["asr_safe"] is False


def test_backdoor_guarded_selection_rejects_when_asr_only_ties():
    from meta_sg.scripts.evaluate_meta_sg_direct import _backdoor_guarded_selection_decision

    decision = _backdoor_guarded_selection_decision(
        base_clean_acc=0.92,
        adapted_clean_acc=0.93,
        base_backdoor_acc=0.90,
        adapted_backdoor_acc=0.90,
        asr_reduction_margin=0.0,
        clean_drop_tolerance=0.02,
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "base"
    assert decision["asr_safe"] is False


def test_backdoor_guarded_selection_rejects_when_clean_drops_too_much():
    from meta_sg.scripts.evaluate_meta_sg_direct import _backdoor_guarded_selection_decision

    decision = _backdoor_guarded_selection_decision(
        base_clean_acc=0.92,
        adapted_clean_acc=0.87,
        base_backdoor_acc=0.90,
        adapted_backdoor_acc=0.70,
        asr_reduction_margin=0.05,
        clean_drop_tolerance=0.02,
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "base"
    assert decision["clean_safe"] is False


def test_parse_paper_online_td3_adaptation_defaults():
    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "paper_online_td3",
        ]
    )

    assert args.few_shot_method == "paper_online_td3"
    assert args.attacker_source == "native"
    assert args.adaptation_attacker_source is None
    assert args.paper_online_windows == 10
    assert args.paper_online_window_horizon == 20
    assert args.paper_online_updates_per_window == 10


def test_parse_allows_zero_attacker_source_for_ablation():
    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "paper_online_td3",
            "--attacker-source",
            "zero",
            "--adaptation-attacker-source",
            "zero",
        ]
    )

    assert args.attacker_source == "zero"
    assert args.adaptation_attacker_source == "zero"


def test_attacker_action_for_source_uses_native_none_by_default():
    from meta_sg.scripts.evaluate_meta_sg_direct import _attacker_action_for_source

    args = parse_args(["--checkpoint", "dummy", "--output-json", "out.json"])

    assert _attacker_action_for_source(args, source="native") is None


def test_attacker_action_for_source_uses_zero_only_when_explicit():
    from meta_sg.scripts.evaluate_meta_sg_direct import _attacker_action_for_source

    args = parse_args(["--checkpoint", "dummy", "--output-json", "out.json"])

    action = _attacker_action_for_source(args, source="zero")

    assert action.shape == (3,)
    assert action.tolist() == [0.0, 0.0, 0.0]


def test_parse_paper_online_backdoor_td3_adaptation_options():
    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "paper_online_backdoor_td3",
            "--backdoor-reward-mode",
            "clean_gated_asr",
            "--backdoor-clean-floor",
            "0.9",
            "--backdoor-reward-lambda",
            "2.0",
            "--backdoor-clean-penalty",
            "3.0",
            "--continuous-window-gate",
            "--continuous-window-clean-drop-tolerance",
            "0.03",
            "--continuous-window-asr-improvement-margin",
            "0.01",
            "--selection-repeats",
            "3",
        ]
    )

    assert args.few_shot_method == "paper_online_backdoor_td3"
    assert args.backdoor_reward_mode == "clean_gated_asr"
    assert args.backdoor_clean_floor == 0.9
    assert args.backdoor_reward_lambda == 2.0
    assert args.backdoor_clean_penalty == 3.0
    assert args.continuous_window_gate is True
    assert args.continuous_window_clean_drop_tolerance == 0.03
    assert args.continuous_window_asr_improvement_margin == 0.01
    assert args.selection_repeats == 3


def test_validation_summary_averages_seed_records():
    from meta_sg.scripts.evaluate_meta_sg_direct import _validation_summary

    summary = _validation_summary(
        [
            {"final_clean_acc": 0.90, "final_backdoor_acc": 0.20, "final_defense_score": 0.70},
            {"final_clean_acc": 0.94, "final_backdoor_acc": 0.10, "final_defense_score": 0.84},
        ]
    )

    assert summary["final_clean_acc"] == pytest.approx(0.92)
    assert summary["final_backdoor_acc"] == pytest.approx(0.15)
    assert summary["final_defense_score"] == pytest.approx(0.77)
    assert summary["num_records"] == 2


def test_backdoor_aware_reward_prefers_low_asr_with_clean_gate():
    from meta_sg.scripts.evaluate_meta_sg_direct import _backdoor_aware_replay_reward

    high_asr = _backdoor_aware_replay_reward(
        {"clean_acc": 0.92, "backdoor_acc": 0.80},
        raw_reward=0.1,
        mode="clean_gated_asr",
        clean_floor=0.90,
        lambda_bd=1.0,
        clean_penalty=2.0,
    )
    low_clean = _backdoor_aware_replay_reward(
        {"clean_acc": 0.82, "backdoor_acc": 0.10},
        raw_reward=0.1,
        mode="clean_gated_asr",
        clean_floor=0.90,
        lambda_bd=1.0,
        clean_penalty=2.0,
    )
    low_asr = _backdoor_aware_replay_reward(
        {"clean_acc": 0.92, "backdoor_acc": 0.10},
        raw_reward=0.1,
        mode="clean_gated_asr",
        clean_floor=0.90,
        lambda_bd=1.0,
        clean_penalty=2.0,
    )

    assert low_asr > high_asr
    assert low_asr > low_clean


def test_continuous_backdoor_window_gate_rejects_bad_window():
    from meta_sg.scripts.evaluate_meta_sg_direct import _continuous_backdoor_window_gate_decision

    accepted = _continuous_backdoor_window_gate_decision(
        previous_clean_acc=0.92,
        current_clean_acc=0.91,
        previous_backdoor_acc=0.80,
        current_backdoor_acc=0.70,
        clean_drop_tolerance=0.02,
        asr_improvement_margin=0.05,
    )
    rejected = _continuous_backdoor_window_gate_decision(
        previous_clean_acc=0.92,
        current_clean_acc=0.91,
        previous_backdoor_acc=0.80,
        current_backdoor_acc=0.79,
        clean_drop_tolerance=0.02,
        asr_improvement_margin=0.05,
    )

    assert accepted["accepted"] is True
    assert accepted["selected"] == "current"
    assert rejected["accepted"] is False
    assert rejected["selected"] == "previous"


def test_continuous_backdoor_window_gate_rejects_asr_tie():
    from meta_sg.scripts.evaluate_meta_sg_direct import _continuous_backdoor_window_gate_decision

    decision = _continuous_backdoor_window_gate_decision(
        previous_clean_acc=0.92,
        current_clean_acc=0.93,
        previous_backdoor_acc=0.80,
        current_backdoor_acc=0.80,
        clean_drop_tolerance=0.02,
        asr_improvement_margin=0.0,
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "previous"
    assert decision["asr_safe"] is False


def test_parse_paper_online_proxy_td3_adaptation_options():
    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--few-shot",
            "--few-shot-method",
            "paper_online_proxy_td3",
            "--proxy-reward-mode",
            "clean_proxy_backdoor",
            "--proxy-update-anomaly-weight",
            "0.25",
            "--proxy-server-lr-weight",
            "0.5",
            "--proxy-synthetic-trigger-weight",
            "2.0",
            "--proxy-synthetic-trigger-patterns",
            "corner_square,center_square",
            "--proxy-synthetic-trigger-targets",
            "all",
            "--proxy-synthetic-trigger-max-batches",
            "3",
            "--proxy-server-lr-offset-step",
            "0.5",
            "--proxy-server-lr-offset-max-steps",
            "4",
        ]
    )

    assert args.few_shot_method == "paper_online_proxy_td3"
    assert args.proxy_reward_mode == "clean_proxy_backdoor"
    assert args.proxy_update_anomaly_weight == 0.25
    assert args.proxy_server_lr_weight == 0.5
    assert args.proxy_synthetic_trigger_weight == 2.0
    assert args.proxy_synthetic_trigger_patterns == "corner_square,center_square"
    assert args.proxy_synthetic_trigger_targets == "all"
    assert args.proxy_synthetic_trigger_max_batches == 3
    assert args.proxy_server_lr_offset_step == 0.5
    assert args.proxy_server_lr_offset_max_steps == 4


def test_proxy_server_lr_offset_targets_fourth_both_action():
    from meta_sg.scripts.evaluate_meta_sg_direct import _proxy_server_lr_offset

    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--defender-third-action",
            "both",
            "--proxy-server-lr-offset-step",
            "0.5",
            "--proxy-server-lr-offset-max-steps",
            "4",
        ]
    )

    offset = _proxy_server_lr_offset(args, act_dim=4)

    assert offset.tolist() == pytest.approx([0.0, 0.0, 0.0, -2.0])


def test_proxy_server_lr_offset_targets_third_server_lr_action():
    from meta_sg.scripts.evaluate_meta_sg_direct import _proxy_server_lr_offset

    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--defender-third-action",
            "server_lr",
            "--proxy-server-lr-offset-step",
            "0.25",
            "--proxy-server-lr-offset-max-steps",
            "2",
        ]
    )

    offset = _proxy_server_lr_offset(args, act_dim=3)

    assert offset.tolist() == pytest.approx([0.0, 0.0, -0.5])


def test_proxy_server_lr_offset_is_disabled_by_default():
    from meta_sg.scripts.evaluate_meta_sg_direct import _proxy_server_lr_offset

    args = parse_args(["--checkpoint", "dummy", "--output-json", "out.json"])

    offset = _proxy_server_lr_offset(args, act_dim=4)

    assert offset.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_proxy_update_anomaly_uses_unlabeled_update_dispersion():
    from meta_sg.scripts.evaluate_meta_sg_direct import _proxy_update_anomaly_score

    quiet = _proxy_update_anomaly_score(
        {"benign_update_norms": [1.0, 1.1], "malicious_update_norms": [0.9, 1.0]}
    )
    spiky = _proxy_update_anomaly_score(
        {"benign_update_norms": [1.0, 1.1], "malicious_update_norms": [8.0, 9.0]}
    )

    assert quiet >= 0.0
    assert spiky > quiet


def test_proxy_replay_reward_does_not_consume_true_backdoor_acc():
    from meta_sg.scripts.evaluate_meta_sg_direct import _proxy_replay_reward

    args = parse_args(
        [
            "--checkpoint",
            "dummy",
            "--output-json",
            "out.json",
            "--proxy-reward-mode",
            "clean_update_anomaly",
            "--proxy-update-anomaly-weight",
            "0.2",
        ]
    )
    base_info = {
        "clean_acc": 0.90,
        "benign_update_norms": [1.0, 1.1],
        "malicious_update_norms": [8.0, 9.0],
    }

    low_true_asr = _proxy_replay_reward(dict(base_info, backdoor_acc=0.0), args=args)
    high_true_asr = _proxy_replay_reward(dict(base_info, backdoor_acc=1.0), args=args)

    assert low_true_asr["reward"] == pytest.approx(high_true_asr["reward"])
    assert "backdoor_acc" not in low_true_asr["diagnostics"]


def test_apply_synthetic_trigger_batch_changes_requested_patch_only():
    import torch

    from meta_sg.scripts.evaluate_meta_sg_direct import _apply_synthetic_trigger_batch

    images = torch.zeros(2, 1, 6, 6)
    triggered = _apply_synthetic_trigger_batch(
        images,
        pattern="corner_square",
        trigger_size=2,
        trigger_value=3.0,
    )

    assert torch.all(images == 0)
    assert torch.all(triggered[:, :, -2:, -2:] == 3.0)
    assert torch.all(triggered[:, :, :4, :4] == 0.0)


def test_synthetic_trigger_proxy_uses_candidate_trigger_not_true_asr():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from meta_sg.scripts.evaluate_meta_sg_direct import _synthetic_trigger_proxy_backdoor_acc

    class CornerTargetModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 3, device=x.device)
            activated = x[:, :, -2:, -2:].mean(dim=(1, 2, 3)) > 0.5
            logits[:, 0] = 1.0
            logits[activated, 2] = 5.0
            return logits

    loader = DataLoader(TensorDataset(torch.zeros(4, 1, 6, 6), torch.zeros(4, dtype=torch.long)), batch_size=2)

    result = _synthetic_trigger_proxy_backdoor_acc(
        CornerTargetModel(),
        loader,
        device=torch.device("cpu"),
        target_classes=[2],
        patterns=["corner_square"],
        trigger_size=2,
        trigger_value=1.0,
        max_batches=1,
    )

    assert result["proxy_backdoor_acc"] == pytest.approx(1.0)
    assert result["target_class"] == 2
    assert result["pattern"] == "corner_square"
