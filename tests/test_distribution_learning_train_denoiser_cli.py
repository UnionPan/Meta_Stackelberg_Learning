from fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.train_denoiser import (
    parse_args,
)


def test_distribution_learning_train_denoiser_defaults_to_phase1_simulator_asset():
    args = parse_args([])

    assert args.output == "fl_sandbox/assets/autoencoder_mnist.pt"
    assert args.seed_samples == 200
    assert args.train_samples == 180
    assert args.noise_std == 0.3
    assert args.batch_size == 32
    assert args.dataset == "mnist"


def test_distribution_learning_train_denoiser_accepts_smoke_overrides():
    args = parse_args(
        [
            "--output",
            "tmp/denoiser.pt",
            "--epochs",
            "1",
            "--seed-samples",
            "8",
            "--train-samples",
            "6",
            "--noise-std",
            "0.2",
            "--device",
            "cpu",
        ]
    )

    assert args.output == "tmp/denoiser.pt"
    assert args.epochs == 1
    assert args.seed_samples == 8
    assert args.train_samples == 6
    assert args.noise_std == 0.2
    assert args.device == "cpu"
