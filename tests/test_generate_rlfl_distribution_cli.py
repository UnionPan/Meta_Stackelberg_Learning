from fl_sandbox.scripts.generate_rlfl_distribution import build_run_config, parse_args


def test_parse_args_defaults_to_paper_distribution_learning_parameters():
    args = parse_args([])

    assert args.rounds == 100
    assert args.num_clients == 100
    assert args.num_attackers == 20
    assert args.subsample_rate == 0.1
    assert args.batch_size == 128
    assert args.noniid_q == 0.1
    assert args.seed_samples == 200
    assert args.reconstruction_batch_size == 32
    assert args.max_iterations == 10000
    assert args.denoiser_mode == "h5"
    assert args.autoencoder_path.endswith("fl_sandbox/assets/autoencoder_mnist.h5")


def test_build_run_config_allows_smoke_overrides_from_cli():
    args = parse_args(
        [
            "--rounds",
            "3",
            "--num-clients",
            "12",
            "--num-attackers",
            "4",
            "--subsample-rate",
            "0.25",
            "--batch-size",
            "16",
            "--noniid-q",
            "0.3",
            "--seed",
            "99",
        ]
    )

    config = build_run_config(args)

    assert config.data.dataset == "mnist"
    assert config.data.split_mode == "paper_q"
    assert config.data.noniid_q == 0.3
    assert config.fl.num_clients == 12
    assert config.fl.num_attackers == 4
    assert config.fl.subsample_rate == 0.25
    assert config.runtime.batch_size == 16
    assert config.runtime.seed == 99
    assert config.defender.type == "clipped_median"
