import csv
import json

import numpy as np
import torch
from PIL import Image

from fl_sandbox.attacks.rl_attacker.proxy.paper_dataset import (
    PaperDistributionDataset,
    PaperDistributionSampler,
)


def _write_distribution(root, *, count=5):
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    with (root / "data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for idx in range(count):
            writer.writerow([idx, idx % 3])
            image = np.full((28, 28), (idx * 20) % 256, dtype=np.uint8)
            Image.fromarray(image, mode="L").save(train_dir / f"{idx}.png")
    (root / "metadata.json").write_text(
        json.dumps({"mean": [0.5], "std": [0.25]}),
        encoding="utf-8",
    )


def test_paper_distribution_dataset_loads_phase1_artifact(tmp_path):
    _write_distribution(tmp_path, count=5)

    dataset = PaperDistributionDataset(tmp_path)

    assert len(dataset) == 5
    assert dataset.num_classes == 3
    images, labels = dataset.sample(3, torch.device("cpu"))
    assert images.shape == (3, 1, 28, 28)
    assert images.dtype == torch.float32
    assert labels.shape == (3,)
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all()
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 2


def test_paper_distribution_sampler_supports_uniform_and_growth(tmp_path):
    _write_distribution(tmp_path, count=320)
    dataset = PaperDistributionDataset(tmp_path)

    uniform = PaperDistributionSampler(dataset, growth_mode="full")
    growth = PaperDistributionSampler(dataset, growth_mode="paper_growth")

    assert len(uniform) == 320
    assert len(growth) == 200
    growth.advance_episode()
    assert len(growth) == 280
    growth.advance_episode()
    assert len(growth) == 320

    images, labels = growth.sample(16, torch.device("cpu"))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)
