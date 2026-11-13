# CIFAR-10

This package groups dataset loading, CNN and ResNet classifiers, and normalized backdoor triggers.
CIFAR-10 has 50,000 training and 10,000 test RGB images, with ten classes and 32×32 pixels per image.

[data.py](data.py) uses Torchvision's official splits under `data_root/cifar-10-batches-py/`.
Downloading is opt-in. Images become `float32 [3, 32, 32]` tensors normalized with mean and standard deviation `(0.5, 0.5, 0.5)`.
Optional training augmentation uses a padded random crop and horizontal flip; evaluation is deterministic.

[model.py](model.py) preserves `CIFAR10CNN` and its Xavier weight/zero bias initialization:

```text
RGB 32×32 -> Conv 3×3, 32 -> ReLU -> MaxPool 2×2
          -> Conv 3×3, 64 -> ReLU -> MaxPool 2×2
          -> Conv 3×3, 64 -> ReLU -> Flatten 4096
          -> Linear 128 -> ReLU -> Linear 10 logits
```

All convolutions use padding 1. The model has no BatchNorm buffers, matching the shared parameter-delta protocol.
[triggers.py](triggers.py) preserves the four existing black-pixel DBA patterns; selecting no patch index composes their union for ASR.
The same complete trigger supports source-specific BFL training.

[resnet.py](resnet.py) preserves the existing CIFAR ResNet-18 recipe: 3×3 stride-one stem with padding 1, no initial maxpool, BatchNorm, and ten output classes. No pretrained weights are loaded.
The existing CIFAR attack/defense matrices use `CIFAR10CNN`; separate clean FedAvg experiments used this ResNet architecture.
Use `"model": {"name": "resnet18"}` to select it. The loader's normalization and augmentation remain explicit dataset options.
See [the ResNet example](../../../../configs/cifar10_resnet18.json) and [buffer synchronization](../../federated/README.md#batchnorm-buffers).

Use [the CIFAR-10 JSON](../../../../configs/cifar10.json) with the [shared entry point](../../experiments/README.md).
