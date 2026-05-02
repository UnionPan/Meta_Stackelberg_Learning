# meta_sg Architecture

This package is the paper-aligned Meta-Stackelberg learning stack. It is
intentionally independent from `fl_sandbox` for now; the real FL adapter should
later implement `simulation.interface.FLCoordinator`.

## Layers

### 1. Simulation Core

Path: `meta_sg/simulation/`

Responsibility: execute the FL transition only.

```text
W_t -> W_{t+1}
```

The simulation layer owns global weights, client sampling, local training,
malicious-client slots, reset/snapshot, and lightweight round summaries.

Important paper constraint: post-training defenses such as NeuroClip and Prun
are reward/evaluation operations on a copy of the global model. They must not
change the next FL state during ordinary rounds.

### 2. Strategy Domain

Path: `meta_sg/strategies/`

Responsibility: define attack/defense semantics and action decoding.

The defender action follows the paper:

```text
a_D^t = (alpha_t, beta_t, epsilon_t / sigma_t)
```

where alpha is norm bounding, beta is coordinate-wise trimmed mean, and
epsilon/sigma configures the post-training defense used for reward/evaluation.

Fixed attacks such as IPM/LMP/BFL/DBA are pre-defined methods. Adaptive attacks
such as RL/BRL own trainable attacker policies and participate in the
best-response inner loop.

### 3. BSMG Game Environment

Path: `meta_sg/games/`

Responsibility: expose one FL round as one game step.

```text
raw policy action -> decision -> FL round -> reward -> next observation
```

The state is the compressed global model, matching the paper's `s_t = W_t`.
The current implementation compresses the tail layers of the weight list.

### 4. Learning System

Path: `meta_sg/learning/`

Responsibility: collect H-step trajectories, reuse them through TD3 replay
buffers, train adaptive attackers toward best responses, and update the
defender meta-policy with a Reptile-style meta update.

The expensive operation is collecting FL trajectories. TD3 updates must reuse
replay-buffer data rather than forcing a fresh H-round FL rollout for every
gradient step.

## Efficiency Rules

- Collect H game steps once, then run multiple TD3 updates from replay buffers.
- Use fixed rollout policies for non-adaptive attacks; do not accidentally turn
  IPM/LMP/BFL/DBA into learned attackers.
- Keep `env.reset()` cheap: restore weights/RNG/history, not datasets.
- Use `eval_every` to avoid full clean/backdoor evaluation on every training
  step.
- Keep each attack task rollout independent so K sampled attacks can later run
  in parallel.

## Paper-Scale Defaults

```text
pre-training:
  T = 100
  K = 10
  H = 200 for MNIST, 500 for CIFAR-10
  l = N_A = N_D = 10

online adaptation:
  H = 100 for MNIST, 200 for CIFAR-10
  l = 10
```

For development, use much smaller settings first, for example `T=2`, `K=1`,
`H=3`, `l=1`, `N_A=1`.
