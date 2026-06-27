"""Tests for extracting physical-target oracle labels from direct-eval JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _candidate(
    label: str,
    *,
    q_clean: float,
    q_asr: float,
    q_score: float,
    d_clean: float,
    d_asr: float,
    d_score: float,
    alpha: float | None = None,
    end_round: int | None = None,
) -> dict:
    return {
        "offset_label": label,
        "target_alpha": alpha,
        "target_beta": None if alpha is None else 0.38,
        "target_start_round": None if alpha is None else 0,
        "target_end_round": end_round,
        "offset": [0.0, 0.0, 0.0],
        "query_clean_mean": q_clean,
        "query_backdoor_mean": q_asr,
        "query_score_mean": q_score,
        "query_records": [
            {
                "final_clean_acc": q_clean,
                "final_backdoor_acc": q_asr,
                "final_defense_score": q_score,
            }
        ],
        "deployment_record": {
            "final_clean_acc": d_clean,
            "final_backdoor_acc": d_asr,
            "final_defense_score": d_score,
        },
        "deployment_clean_acc": d_clean,
        "deployment_backdoor_acc": d_asr,
        "deployment_defense_score": d_score,
    }


def _row(*, scenario: str, seed: int, candidates: list[dict]) -> dict:
    base = candidates[0]["deployment_record"]
    return {
        "scenario": scenario,
        "attack_type": scenario,
        "seed": seed,
        "final_clean_acc": base["final_clean_acc"],
        "final_backdoor_acc": base["final_backdoor_acc"],
        "final_defense_score": base["final_defense_score"],
        "few_shot_adaptation": {
            "candidate_scores": candidates,
            "selection": {},
        },
    }


def test_physical_target_label_extractor_normalizes_labels():
    from meta_sg.scripts.extract_physical_target_labels import _label_from_offset_label

    assert _label_from_offset_label("zero") == "zero"
    assert _label_from_offset_label("physical_a0.10_b0.38_w0_40") == "a0.10_w40"
    assert _label_from_offset_label("physical_a0.14_b0.38_full") == "a0.14_full"


def test_extract_physical_target_label_samples_replays_relative_drop_oracle(tmp_path: Path):
    from meta_sg.scripts.extract_physical_target_labels import extract_label_samples

    eval_path = tmp_path / "eval.json"
    rows = [
        _row(
            scenario="bfl",
            seed=1112,
            candidates=[
                _candidate(
                    "zero",
                    q_clean=0.93,
                    q_asr=0.70,
                    q_score=-0.47,
                    d_clean=0.94,
                    d_asr=0.70,
                    d_score=-0.46,
                ),
                _candidate(
                    "physical_a0.10_b0.38_w0_40",
                    q_clean=0.91,
                    q_asr=0.02,
                    q_score=0.87,
                    d_clean=0.935,
                    d_asr=0.03,
                    d_score=0.875,
                    alpha=0.10,
                    end_round=40,
                ),
            ],
        )
    ]
    eval_path.write_text(json.dumps(rows), encoding="utf-8")

    samples = extract_label_samples(
        [eval_path],
        asr_reduction_margin=0.005,
        query_clean_floor=0.0,
        query_clean_drop_tolerance=0.04,
        score_slack=0.0,
        deployment_clean_floor=None,
        deployment_clean_drop_tolerance=0.02,
        deployment_asr_ceiling=0.30,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample["label"] == "a0.10_w40"
    assert sample["selected_offset_label"] == "physical_a0.10_b0.38_w0_40"
    assert sample["attack_name"] == "bfl"
    assert sample["seed"] == 1112
    assert sample["oracle_metrics"]["clean_drop"] == pytest.approx(0.005)
    assert sample["oracle_metrics"]["deployment_backdoor_acc"] == pytest.approx(0.03)
    assert sample["constraints"]["deployment_clean_drop_tolerance"] == pytest.approx(0.02)
    assert sample["candidate_query_summaries"][1]["label"] == "a0.10_w40"


def test_extract_physical_target_label_samples_falls_back_to_zero_when_drop_too_large(tmp_path: Path):
    from meta_sg.scripts.extract_physical_target_labels import extract_label_samples

    eval_path = tmp_path / "eval.json"
    rows = [
        _row(
            scenario="rl_backdoor",
            seed=1312,
            candidates=[
                _candidate(
                    "zero",
                    q_clean=0.93,
                    q_asr=0.97,
                    q_score=-1.01,
                    d_clean=0.944,
                    d_asr=0.767,
                    d_score=-0.59,
                ),
                _candidate(
                    "physical_a0.10_b0.38_w0_40",
                    q_clean=0.907,
                    q_asr=0.59,
                    q_score=-0.273,
                    d_clean=0.916,
                    d_asr=0.02,
                    d_score=0.876,
                    alpha=0.10,
                    end_round=40,
                ),
            ],
        )
    ]
    eval_path.write_text(json.dumps(rows), encoding="utf-8")

    sample = extract_label_samples(
        [eval_path],
        asr_reduction_margin=0.005,
        query_clean_floor=None,
        query_clean_drop_tolerance=0.04,
        score_slack=0.0,
        deployment_clean_floor=None,
        deployment_clean_drop_tolerance=0.02,
        deployment_asr_ceiling=0.30,
    )[0]

    assert sample["label"] == "zero"
    assert sample["selection"]["selection_stage"] == "deployment_rejected"
    assert sample["oracle_metrics"]["clean_drop"] == pytest.approx(0.0)
    assert sample["oracle_metrics"]["query_selected_offset_label"] == "physical_a0.10_b0.38_w0_40"


def test_extract_physical_target_labels_script_writes_jsonl_and_metrics(tmp_path: Path):
    from meta_sg.scripts.extract_physical_target_labels import main, parse_args

    eval_path = tmp_path / "eval.json"
    output_path = tmp_path / "labels.jsonl"
    metrics_path = tmp_path / "metrics.json"
    rows = [
        _row(
            scenario="dba",
            seed=1112,
            candidates=[
                _candidate(
                    "zero",
                    q_clean=0.94,
                    q_asr=0.0,
                    q_score=0.94,
                    d_clean=0.94,
                    d_asr=0.0,
                    d_score=0.94,
                )
            ],
        )
    ]
    eval_path.write_text(json.dumps(rows), encoding="utf-8")

    args = parse_args(
        [
            "--input-json",
            str(eval_path),
            "--output-jsonl",
            str(output_path),
            "--metrics-json",
            str(metrics_path),
            "--deployment-clean-drop-tolerance",
            "0.02",
            "--deployment-asr-ceiling",
            "0.30",
        ]
    )
    assert args.deployment_clean_drop_tolerance == pytest.approx(0.02)

    main(
        [
            "--input-json",
            str(eval_path),
            "--output-jsonl",
            str(output_path),
            "--metrics-json",
            str(metrics_path),
            "--deployment-clean-drop-tolerance",
            "0.02",
            "--deployment-asr-ceiling",
            "0.30",
        ]
    )

    samples = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(samples) == 1
    assert samples[0]["label"] == "zero"
    assert metrics["num_samples"] == 1
    assert metrics["label_counts"] == {"zero": 1}
