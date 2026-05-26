# Phase 1 Simulator Preparation

Phase 1 prepares the proxy distribution used by the RL attacker simulator.
It does not train the attack policy directly.

## Distribution Learning

Generate the paper-style proxy dataset:

```bash
python -u -m fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.generate_distribution --device auto
```

Default outputs:

```text
fl_sandbox/outputs/rlfl_distribution_paper/mnist_clipping_median_q_0.1/
  no_process/*.png   raw gradient-inversion reconstructions
  train/*.png        denoised images used by simulator
  data.csv           image index and label
  metadata.json      run configuration and counts
```

The default size is:

```text
200 seed images + 100 rounds * 32 reconstructed images = 3400 images
```

## Denoiser

The denoiser is a Phase 1 simulator asset. It is trained before distribution
learning from clean/noisy MNIST image pairs:

```text
clean MNIST image
  + Gaussian noise, std=0.3
  -> noisy image
  -> denoising autoencoder
  -> clean image target
```

Train a PyTorch checkpoint:

```bash
python -u -m fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.train_denoiser \
  --output fl_sandbox/assets/autoencoder_mnist.pt \
  --epochs 100 \
  --device auto
```

The reproduction script still defaults to the original Keras-compatible
checkpoint:

```text
fl_sandbox/assets/autoencoder_mnist.h5
```

Use the PyTorch fallback denoiser inside distribution learning with:

```bash
python -u -m fl_sandbox.attacks.rl_attacker.simulator.distribution_learning.generate_distribution \
  --denoiser-mode pytorch \
  --denoiser-epochs 100 \
  --device auto
```

## Simulator Consumption

After generation, the simulator reads:

```text
train/*.png + data.csv
```

as image-label pairs through the proxy distribution dataset. `no_process/`
is kept for inspection and debugging; `train/` is the denoised distribution
used for local training in the simulator.
