import torch

from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.generate_distribution import (
    build_run_config,
    parse_args,
    sample_pre_initial_batch,
    sample_pre_initial_images,
)


def test_parse_args_defaults_to_phase1_distribution_learning_parameters():
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


def test_parse_args_accepts_pre_initialization():
    args = parse_args(["--init", "pre"])

    assert args.init == "pre"


def test_sample_pre_initial_images_draws_from_current_proxy_pool():
    pool = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 2, 2)
    generator = torch.Generator().manual_seed(3)

    sampled = sample_pre_initial_images(
        pool,
        num_images=8,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert sampled.shape == (8, 1, 2, 2)
    flattened_pool = {tuple(image.flatten().tolist()) for image in pool}
    for image in sampled:
        assert tuple(image.flatten().tolist()) in flattened_pool


def test_sample_pre_initial_batch_keeps_images_and_labels_aligned():
    pool = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 2, 2)
    labels = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    generator = torch.Generator().manual_seed(4)

    sampled_images, sampled_labels = sample_pre_initial_batch(
        pool,
        labels,
        num_images=8,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert sampled_images.shape == (8, 1, 2, 2)
    assert sampled_labels.shape == (8,)
    for image, label in zip(sampled_images, sampled_labels):
        source_index = int(label.item() - 10)
        assert torch.equal(image.cpu(), pool[source_index])
