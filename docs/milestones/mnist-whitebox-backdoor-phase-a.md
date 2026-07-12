# MNIST White-Box Backdoor Phase A

## Scope

This milestone verifies the real-data execution path for the Meta-SG white-box backdoor setting. It is an
execution and determinism smoke run, not a scientific performance result.

## Data protocol

- Protocol: `mnist-whitebox-real-data-v1`
- Client training set: all 60,000 official MNIST training examples
- Federation: 100 clients, IID partition, 5 malicious clients, 10 sampled clients per round
- Server reward view: 200 stratified read-only examples from the client training set
- Held-out query set: all 10,000 official MNIST test examples
- Query data used for training/replay/checkpoint selection: no
- Trigger task: source digit 1 to target digit 7
- Trigger fixture: `mnist-global-1-to-7-v1`
- Trigger SHA-256: `c5226726b8b70efa2f667d59784a81e6f3f7e1a359a53d067b2db086d570c93b`

## Policy protocol

- Defender action: `(alpha, beta, epsilon)`
- BRL Attacker action: `(rho, eta, E)`
- Both action dimensions: 3
- Algorithm: Meta-SG Algorithm 1 with the practical Reptile leader update
- Attacker response target: the current outer-loop meta Defender (Algorithm 1 line 14)
- Reptile Defender-gradient point: the task-adapted Defender (Algorithm 1 line 17)
- Pretrained attack origins:
  - `brl-norm`: fixed Norm Bounding
  - `brl-neuroclip`: fixed NeuroClip

## Real MNIST micro run

The micro profile ran two FL rounds per BRL pretraining task and one Algorithm 1 iteration with
`K=1`, `N_A=1`, `N_D=1`, and `H=1`. This produced four attack-pretraining FL rounds and three Algorithm 1
support trajectories (Defender adaptation, Attacker response, and leader update).

Artifacts:

- `fl_sandbox/runs/mnist_whitebox_micro/brl_domain.pt`
- `fl_sandbox/runs/mnist_whitebox_micro/brl_domain.pt.manifest.json`
- `fl_sandbox/runs/mnist_whitebox_micro/brl_domain.pt.checkpoints/brl-norm.pt`
- `fl_sandbox/runs/mnist_whitebox_micro/brl_domain.pt.checkpoints/brl-neuroclip.pt`
- `fl_sandbox/runs/mnist_whitebox_micro/meta_sg/manifest.json`
- `fl_sandbox/runs/mnist_whitebox_micro/meta_sg/policies.pt`
- `fl_sandbox/runs/mnist_whitebox_micro/scientific/scientific.json`

The pretraining and Meta-SG manifests contain matching client-partition and trigger hashes. The final micro
Defender fingerprint is:

```text
54c6d469b4cc070fff264610bdef007e47d61c5acd98a252f91096180962b6f0
```

## Verification

```text
933 passed in 77.40s
```

Artifact invariants verified independently:

- 60,000 client training examples;
- 200 reward-view examples;
- 10,000 held-out query examples;
- matching partition hashes between BRL pretraining and Algorithm 1;
- matching trigger hashes;
- Algorithm 1 rather than Algorithm 2;
- three-dimensional Defender and Attacker actions;
- disjoint support and query seeds;
- no query data in training.

## Scientific status

The held-out scientific protocol now evaluates all Meta-SG behavior checks plus clean accuracy, source-class ASR,
safe loss, and target loss on the complete 10,000-example official MNIST test set. The micro result failed without
changing any threshold:

```text
combined Gate: failed
Meta-SG Gate: failed
white-box safety Gate: failed

learned clean accuracy: 0.4045       required >= 0.8
learned ASR:            0.0          required <= 0.2
ASR reduction:          0.0          required >= 0.2
Defender adaptation:    0.0          required >= 0.001
meta margin:            -0.00002717  required >= 0.001
```

The ASR ceiling check passes in isolation, but it is not evidence of a useful defense: the one-round model has low
clean accuracy and no-adaptation ASR is already zero. The micro run therefore proves execution, artifact provenance,
checkpoint/resume, query isolation, and Gate preservation only. A performance claim requires the preregistered
scaled run; no scientific threshold was lowered and no Gate is reported as passed.
