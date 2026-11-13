# Tiny ImageNet

Tiny ImageNet contains 200 classes of 64×64 RGB images: 500 training and 50 labeled validation images per class.
This package evaluates on the labeled validation split; the archive's test images have no supplied labels.

[data.py](data.py) reads `data_root/tiny-imagenet-200.zip` directly, using the official archive layout.
It sorts `wnids.txt` to assign class IDs, reads validation labels from `val_annotations.txt`, and selects a seeded number of images per class.
Image paths remain available as `dataset.paths`; labels are in `dataset.targets`.
Selected images are decoded to CPU `uint8 [N, 3, 64, 64]` storage. The full training and validation image tensors require about 1.35 GB before other process memory.
Set `train_per_class` and `val_per_class` for smaller subsets; the archive itself must be supplied locally.

| Normalization | Mean | Standard deviation |
| --- | --- | --- |
| `legacy` (default) | `(0.5, 0.5, 0.5)` | `(0.5, 0.5, 0.5)` |
| `tiny` | `(0.4802, 0.4481, 0.3975)` | `(0.2770, 0.2691, 0.2821)` |

Each sample is `(float32 [3, 64, 64] image, integer label)`.
Optional training augmentation uses rotation up to 10 degrees, random crop with padding 4, and horizontal flip. Validation preprocessing is deterministic.

[model.py](model.py) preserves the lightweight `TinyImageNetCNN` from the shared framework:

```text
RGB 64×64 -> Conv 3×3, 32 -> ReLU -> MaxPool 2×2
          -> Conv 3×3, 64 -> ReLU -> MaxPool 2×2
          -> Conv 3×3, 64 -> ReLU -> AdaptiveAvgPool 4×4
          -> Flatten 1024 -> Linear 128 -> ReLU -> Linear 200 logits
```

All convolutions use padding 1; weights use Xavier initialization and biases are zero.
The lightweight model has no BatchNorm buffers and comes from the early resource-check branch.

[resnet.py](resnet.py) also provides the standard torchvision ResNet-18 used in the formal Tiny ImageNet LMP, DBA, and BFL experiments: original 7×7 stride-two stem, initial maxpool, BatchNorm, and a 200-class output layer, without pretrained weights.
Select it with `"model": {"name": "resnet18"}`. The [ResNet example](../../../../configs/tiny_imagenet_resnet18.json) explicitly uses `normalization="tiny"` and training augmentation.
The shared entry point supports fixed-step or full-epoch SGD, momentum, weight decay, round-wise learning-rate decay, and [buffer synchronization](../../federated/README.md#batchnorm-buffers).
The [Tiny BFL preset](../../../../configs/tiny_imagenet_bfl.json) shows five local epochs and the decayed optimizer settings; it does not claim identical historical RNG streams or malicious identities.

[triggers.py](triggers.py) implements the later Tiny ImageNet white pattern: four disjoint 2×10 patches, 80 spatial pixels in total.
Trigger values use the selected normalization preset. A patch index selects one group for DBA; no index composes the full trigger for BFL and ASR.
The historical black-trigger BFL variant is a different task and is not selected by this entry point.

Use [the Tiny ImageNet JSON](../../../../configs/tiny_imagenet.json) with the [shared entry point](../../experiments/README.md).
