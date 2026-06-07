import json

from meta_sg.scripts.run_single_attacker_defender_tensorboard import (
    build_bsmg_config,
    main,
    parse_args,
    single_attack_domain,
)


def test_single_attacker_runner_uses_paper_aligned_config():
    cfg = build_bsmg_config(horizon=3, alpha_max=5.0)

    assert cfg.horizon == 3
    assert cfg.num_tail_layers == 2
    assert cfg.history_len == 0
    assert cfg.reward_mode == "loss"
    assert cfg.lambda_bd == 0.0
    assert cfg.action_prior_weight == 0.0
    assert cfg.use_neuroclip is True
    assert cfg.relative_alpha is False


def test_single_attacker_domain_contains_only_rl():
    domain = single_attack_domain()

    assert len(domain) == 1
    assert domain[0].name == "rl"
    assert domain[0].objective == "untargeted"
    assert domain[0].adaptive is True


def test_parse_args_defaults_are_small_and_reproducible():
    args = parse_args(["--run-name", "unit", "--horizon", "3", "--seeds", "1", "2"])

    assert args.run_name == "unit"
    assert args.horizon == 3
    assert args.seeds == [1, 2]
    assert args.alpha_max > 0.0


def test_single_attacker_runner_smoke(tmp_path):
    main(
        [
            "--backend",
            "stub",
            "--run-name",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--td3-iters",
            "1",
            "--td3-updates",
            "1",
            "--hidden-dim",
            "8",
            "--batch-size",
            "2",
            "--buffer-capacity",
            "32",
            "--exploration-noise",
            "0.0",
            "--device",
            "cpu",
        ]
    )

    summary = tmp_path / "smoke" / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["attack_domain"][0]["name"] == "rl"
    assert payload["bsmg_config"]["reward_mode"] == "loss"
