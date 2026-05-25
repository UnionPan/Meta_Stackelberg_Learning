import csv
import json

import h5py
import numpy as np
import torch
from torch import nn

from fl_sandbox.attacks.rl_attacker.paper_distribution import (
    KerasMnistAutoencoder,
    PaperGradientReconstructor,
    ReconstructorConfig,
    estimate_aggregate_gradient,
    load_keras_mnist_autoencoder,
    write_distribution_artifacts,
)


def test_estimate_aggregate_gradient_uses_previous_minus_current_over_lr_gap():
    previous = [np.asarray([3.0, 5.0], dtype=np.float32)]
    current = [np.asarray([1.0, 1.0], dtype=np.float32)]

    gradient = estimate_aggregate_gradient(
        previous,
        current,
        lr=0.5,
        round_gap=4,
        device=torch.device("cpu"),
    )

    assert torch.allclose(gradient[0], torch.tensor([1.0, 2.0]))


def test_write_distribution_artifacts_creates_pngs_csv_and_metadata(tmp_path):
    images = torch.stack(
        [
            torch.zeros((1, 4, 4), dtype=torch.float32),
            torch.full((1, 4, 4), 0.5, dtype=torch.float32),
        ]
    )
    labels = torch.tensor([3, 7], dtype=torch.long)

    result = write_distribution_artifacts(
        images=images,
        labels=labels,
        output_dir=tmp_path,
        mean=(0.0,),
        std=(1.0,),
        metadata={"source": "unit-test"},
    )

    assert result.num_images == 2
    assert (tmp_path / "no_process" / "0.png").is_file()
    assert (tmp_path / "no_process" / "1.png").is_file()
    assert (tmp_path / "train" / "0.png").is_file()
    assert (tmp_path / "train" / "1.png").is_file()

    with (tmp_path / "data.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows == [["0", "3"], ["1", "7"]]

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["num_images"] == 2
    assert metadata["source"] == "unit-test"


def test_write_distribution_artifacts_removes_stale_pngs(tmp_path):
    stale_no_process = tmp_path / "no_process"
    stale_train = tmp_path / "train"
    stale_no_process.mkdir()
    stale_train.mkdir()
    (stale_no_process / "stale.png").write_bytes(b"old")
    (stale_train / "stale.png").write_bytes(b"old")

    images = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)

    write_distribution_artifacts(
        images=images,
        labels=labels,
        output_dir=tmp_path,
        mean=(0.0,),
        std=(1.0,),
    )

    assert not (stale_no_process / "stale.png").exists()
    assert not (stale_train / "stale.png").exists()


def test_gradient_reconstructor_returns_bounded_images_and_label_logits():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    images = torch.rand((2, 1, 2, 2), dtype=torch.float32)
    label_logits = torch.randn((2, 2), dtype=torch.float32)
    soft_labels = torch.softmax(label_logits, dim=-1)
    loss = torch.mean(
        torch.sum(-soft_labels * torch.log_softmax(model(images), dim=-1), dim=-1)
    )
    observed_gradient = torch.autograd.grad(loss, model.parameters())

    reconstructor = PaperGradientReconstructor(
        model=model,
        config=ReconstructorConfig(
            max_iterations=2,
            lr=0.05,
            total_variation=0.0,
            init="zeros",
        ),
        mean=(0.0,),
        std=(1.0,),
        num_images=2,
        image_shape=(1, 2, 2),
        device=torch.device("cpu"),
    )

    result = reconstructor.reconstruct(observed_gradient, labels=None)

    assert result.images.shape == (2, 1, 2, 2)
    assert result.label_logits.shape == (2, 2)
    assert torch.isfinite(result.images).all()
    assert torch.min(result.images).item() >= 0.0
    assert torch.max(result.images).item() <= 1.0


def test_load_keras_mnist_autoencoder_reads_h5_weights(tmp_path):
    h5_path = tmp_path / "autoencoder_mnist.h5"
    with h5py.File(h5_path, "w") as handle:
        root = handle.create_group("model_weights")
        shapes = {
            "conv2d": ((3, 3, 1, 32), (32,)),
            "conv2d_1": ((3, 3, 32, 32), (32,)),
            "conv2d_2": ((3, 3, 32, 32), (32,)),
            "conv2d_3": ((3, 3, 32, 32), (32,)),
            "conv2d_4": ((3, 3, 32, 1), (1,)),
        }
        for layer, (kernel_shape, bias_shape) in shapes.items():
            group = root.create_group(layer).create_group(layer)
            group.create_dataset("kernel:0", data=np.zeros(kernel_shape, dtype=np.float32))
            group.create_dataset("bias:0", data=np.zeros(bias_shape, dtype=np.float32))

    model = load_keras_mnist_autoencoder(h5_path)
    output = model(torch.zeros((1, 1, 28, 28), dtype=torch.float32))

    assert isinstance(model, KerasMnistAutoencoder)
    assert output.shape == (1, 1, 28, 28)
    assert torch.allclose(output, torch.full_like(output, 0.5), atol=1e-6)
