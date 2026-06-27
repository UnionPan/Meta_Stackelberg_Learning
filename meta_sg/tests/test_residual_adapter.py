"""Tests for the trained context-conditioned residual adapter."""

from pathlib import Path

import numpy as np
import pytest
import torch


def test_encode_residual_features_uses_support_metrics_and_attack_context():
    from meta_sg.learning.residual_adapter import encode_residual_features

    support_summary = {
        "support_score_mean": 0.81,
        "support_clean_mean": 0.83,
        "support_backdoor_mean": 0.02,
        "support_reward_mean": 0.77,
        "support_score_slope": 0.04,
        "support_clean_slope": 0.03,
        "support_backdoor_slope": -0.01,
        "support_reward_slope": 0.02,
        "direction": [0.04, 0.01, 0.03],
        "normalized_direction": [0.78446454, 0.19611613, 0.5883484],
        "direction_norm": 0.0509902,
    }

    features, schema = encode_residual_features(
        support_summary,
        attack_name="bfl",
        attack_objective="targeted",
        act_dim=3,
    )

    assert features.dtype == np.float32
    assert len(features) == len(schema)
    assert schema[:4] == [
        "support_score_mean",
        "support_clean_mean",
        "support_backdoor_mean",
        "support_reward_mean",
    ]
    assert features[schema.index("objective_targeted")] == pytest.approx(1.0)
    assert features[schema.index("objective_untargeted")] == pytest.approx(0.0)
    assert features[schema.index("attack_bfl")] == pytest.approx(1.0)
    assert features[schema.index("attack_ipm")] == pytest.approx(0.0)
    assert features[schema.index("direction_2")] == pytest.approx(0.03)
    assert features[schema.index("normalized_direction_1")] == pytest.approx(0.19611613)


def test_residual_adapter_training_learns_synthetic_offsets():
    from meta_sg.learning.residual_adapter import fit_residual_adapter, predict_residual_offset

    samples = []
    for idx, slope in enumerate([-0.04, -0.02, 0.02, 0.04]):
        samples.append(
            {
                "support_summary": {
                    "support_score_mean": 0.8,
                    "support_clean_mean": 0.8,
                    "support_backdoor_mean": 0.01,
                    "support_reward_mean": 0.7,
                    "support_score_slope": slope,
                    "support_clean_slope": slope,
                    "support_backdoor_slope": 0.0,
                    "support_reward_slope": slope,
                    "direction": [slope, 0.0],
                    "normalized_direction": [1.0 if slope > 0 else -1.0, 0.0],
                    "direction_norm": abs(slope),
                },
                "attack_name": "bfl" if idx % 2 == 0 else "dba",
                "attack_objective": "targeted",
                "selected_offset": [0.1 if slope > 0 else -0.1, 0.0],
                "accepted": True,
            }
        )

    adapter = fit_residual_adapter(samples, act_dim=2, bound=0.2, hidden_dim=16, epochs=250, lr=0.03, seed=7)
    positive = predict_residual_offset(adapter, samples[-1], bound=0.2)
    negative = predict_residual_offset(adapter, samples[0], bound=0.2)

    assert positive[0] > 0.05
    assert negative[0] < -0.05
    assert abs(float(positive[1])) < 0.05
    assert abs(float(negative[1])) < 0.05


def test_residual_adapter_prediction_is_clipped_to_bound():
    from meta_sg.learning.residual_adapter import ResidualAdapter, ResidualAdapterNet, predict_residual_offset

    net = ResidualAdapterNet(input_dim=2, act_dim=2, hidden_dim=4)
    with torch.no_grad():
        for param in net.parameters():
            param.zero_()
        net.output.bias[:] = torch.tensor([10.0, -10.0])
    adapter = ResidualAdapter(
        model=net,
        feature_schema=["support_score_mean", "objective_targeted"],
        act_dim=2,
        bound=0.25,
    )
    sample = {
        "support_summary": {"support_score_mean": 0.1},
        "attack_name": "bfl",
        "attack_objective": "targeted",
    }

    offset = predict_residual_offset(adapter, sample, bound=0.25)

    assert offset.tolist() == pytest.approx([0.25, -0.25])


def test_residual_adapter_checkpoint_roundtrip(tmp_path: Path):
    from meta_sg.learning.residual_adapter import (
        fit_residual_adapter,
        load_residual_adapter,
        predict_residual_offset,
        save_residual_adapter,
    )

    sample = {
        "support_summary": {
            "support_score_mean": 0.8,
            "support_clean_mean": 0.8,
            "support_backdoor_mean": 0.01,
            "support_reward_mean": 0.7,
            "support_score_slope": 0.02,
            "support_clean_slope": 0.02,
            "support_backdoor_slope": 0.0,
            "support_reward_slope": 0.02,
            "direction": [0.02, 0.0],
            "normalized_direction": [1.0, 0.0],
            "direction_norm": 0.02,
        },
        "attack_name": "bfl",
        "attack_objective": "targeted",
        "selected_offset": [0.12, -0.02],
        "accepted": True,
    }
    adapter = fit_residual_adapter([sample], act_dim=2, bound=0.2, hidden_dim=8, epochs=150, lr=0.05, seed=3)
    path = tmp_path / "adapter.pt"

    save_residual_adapter(adapter, path, metrics={"train_loss": 0.01})
    loaded, metrics = load_residual_adapter(path)

    assert metrics["train_loss"] == pytest.approx(0.01)
    assert loaded.feature_schema == adapter.feature_schema
    assert loaded.act_dim == adapter.act_dim
    assert predict_residual_offset(loaded, sample, bound=0.2).shape == (2,)


def test_train_residual_adapter_script_trains_checkpoint_from_jsonl(tmp_path: Path):
    import json

    from meta_sg.learning.residual_adapter import load_residual_adapter
    from meta_sg.scripts.train_residual_adapter import main, parse_args

    samples_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "adapter.pt"
    metrics_path = tmp_path / "metrics.json"
    samples = [
        {
            "support_summary": {
                "support_score_mean": 0.8,
                "support_clean_mean": 0.8,
                "support_backdoor_mean": 0.01,
                "support_reward_mean": 0.7,
                "support_score_slope": 0.02,
                "support_clean_slope": 0.02,
                "support_backdoor_slope": 0.0,
                "support_reward_slope": 0.02,
                "direction": [0.02, 0.0],
                "normalized_direction": [1.0, 0.0],
                "direction_norm": 0.02,
            },
            "attack_name": "bfl",
            "attack_objective": "targeted",
            "selected_offset": [0.1, 0.0],
            "accepted": True,
        },
        {
            "support_summary": {
                "support_score_mean": 0.8,
                "support_clean_mean": 0.8,
                "support_backdoor_mean": 0.01,
                "support_reward_mean": 0.7,
                "support_score_slope": -0.02,
                "support_clean_slope": -0.02,
                "support_backdoor_slope": 0.0,
                "support_reward_slope": -0.02,
                "direction": [-0.02, 0.0],
                "normalized_direction": [-1.0, 0.0],
                "direction_norm": 0.02,
            },
            "attack_name": "dba",
            "attack_objective": "targeted",
            "selected_offset": [-0.1, 0.0],
            "accepted": True,
        },
    ]
    samples_path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    args = parse_args(
        [
            "--input-jsonl",
            str(samples_path),
            "--output-checkpoint",
            str(output_path),
            "--metrics-json",
            str(metrics_path),
            "--act-dim",
            "2",
            "--bound",
            "0.2",
            "--hidden-dim",
            "8",
            "--epochs",
            "80",
            "--lr",
            "0.05",
            "--seed",
            "11",
        ]
    )
    assert args.act_dim == 2
    assert args.bound == pytest.approx(0.2)

    main(
        [
            "--input-jsonl",
            str(samples_path),
            "--output-checkpoint",
            str(output_path),
            "--metrics-json",
            str(metrics_path),
            "--act-dim",
            "2",
            "--bound",
            "0.2",
            "--hidden-dim",
            "8",
            "--epochs",
            "80",
            "--lr",
            "0.05",
            "--seed",
            "11",
        ]
    )

    adapter, metrics = load_residual_adapter(output_path)
    written_metrics = json.loads(metrics_path.read_text())
    assert output_path.exists()
    assert metrics_path.exists()
    assert adapter.act_dim == 2
    assert metrics["num_samples"] == 2
    assert written_metrics["num_samples"] == 2
