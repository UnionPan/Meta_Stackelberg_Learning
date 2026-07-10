import pytest

from meta_sg.scripts.evaluate_whitebox_backdoor_selector import (
    Candidate,
    parse_eps_candidates,
    parse_pruning_candidates,
    select_whitebox_candidate,
)


def test_parse_eps_candidates_includes_base_and_fixed_eps_values():
    candidates = parse_eps_candidates("base,1,2.5")

    assert candidates == [
        Candidate(name="base", fixed_neuroclip_epsilon=None),
        Candidate(name="neuroclip_eps_1", fixed_neuroclip_epsilon=1.0),
        Candidate(name="neuroclip_eps_2.5", fixed_neuroclip_epsilon=2.5),
    ]


def test_parse_pruning_candidates_uses_model_aware_pruning_mode():
    candidates = parse_pruning_candidates("0.1,0.25")

    assert candidates == [
        Candidate(name="pruning_0.1", post_defense_mode="model_aware_pruning", fixed_pruning_mask_rate=0.1),
        Candidate(name="pruning_0.25", post_defense_mode="model_aware_pruning", fixed_pruning_mask_rate=0.25),
    ]


def test_whitebox_selector_picks_lowest_asr_candidate_above_clean_floor():
    records = [
        {"candidate": "base", "scenario": "bfl", "final_clean_acc": 0.94, "final_backdoor_acc": 0.91},
        {"candidate": "eps_2", "scenario": "bfl", "final_clean_acc": 0.91, "final_backdoor_acc": 0.12},
        {"candidate": "eps_1", "scenario": "bfl", "final_clean_acc": 0.84, "final_backdoor_acc": 0.02},
    ]

    selected = select_whitebox_candidate(records, clean_floor=0.90)

    assert selected["candidate"] == "eps_2"
    assert selected["final_backdoor_acc"] == pytest.approx(0.12)


def test_whitebox_selector_falls_back_to_cleanest_when_no_candidate_meets_floor():
    records = [
        {"candidate": "base", "scenario": "bfl", "final_clean_acc": 0.88, "final_backdoor_acc": 0.91},
        {"candidate": "eps_1", "scenario": "bfl", "final_clean_acc": 0.84, "final_backdoor_acc": 0.02},
    ]

    selected = select_whitebox_candidate(records, clean_floor=0.90)

    assert selected["candidate"] == "base"
    assert selected["selection_reason"] == "no_clean_safe_candidate"
