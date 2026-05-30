# Meta-SG Co-Learning Design

## Status

Design spec. Implementation has not started.

This design turns `meta_sg` into a co-learning system for an RL poisoning
attacker and a meta-RL defender. It uses Henger Li's "Learning to Attack
Federated Learning" as a reference for attacker training semantics, but keeps
the implementation aligned with this repository's `meta_sg` architecture.

## Goal

Train a defender and attacker that co-evolve toward a Bayesian Stackelberg
game:

- The defender is the leader. It outputs aggregation and robustness parameters
  for each FL round.
- The attacker is the follower. Given the current frozen defender, it trains or
  fine-tunes a best response.
- The outer loop alternates between attacker best-response updates and defender
  meta updates.

The first implementation milestone is not full co-learning. It is a small
closed loop that proves the attacker responds to defender changes.

## Current Repo Context

The repository already has the right high-level pieces:

- `meta_sg/learning/meta_sg_trainer.py` has a Reptile-style defender outer loop.
- `meta_sg/learning/best_response.py` updates adaptive attacker policies.
- `meta_sg/learning/task_runner.py` collects trajectories and coordinates
  defender and attacker updates.
- `meta_sg/games/bsmg_env.py` wraps one FL round as one game step.
- `meta_sg/simulation/fl_sandbox_adapter.py` connects the game environment to
  `fl_sandbox`.
- `fl_sandbox/attacks/rl_attacker/` contains the paper-style TD3 attacker and
  Phase-1 proxy distribution support.

The current `meta_sg` implementation is still a simplified paper-shaped stack.
Both players use the same observation/action dimensions, the adaptive attacker
is IPM-style rather than Henger-style local search, and there is no explicit
response-curve validation that the attacker policy changes when the defender
changes.

## Core Design

### Training Structure

Use iterative best response before attempting nested meta-gradients.

For each outer iteration:

1. Stage A: freeze the defender and warm-start train the attacker toward the
   best response to that defender.
2. Stage B: freeze the attacker and update the defender meta-policy over one or
   more sampled FL conditions.
3. Periodically evaluate response curves, adaptation speed, and held-out
   condition performance.

This separates the fast follower update from the slow leader update. It also
gives clear debugging points: if Stage A cannot produce a changing best
response, Stage B should not be trusted.

### Attacker

The first co-learning attacker should follow the Henger-style clipped-median
attacker:

- Action: 2D continuous vector in `[-1, 1]`.
- Decode:
  - `epsilon = a[0] * 14.9 + 15`, range `[0.1, 29.9]`.
  - `local_steps = a[1] * 24 + 25`, integer range `[1, 49]`.
- Observation:
  - normalized tail or last-layer global weights,
  - number of malicious clients selected this round,
  - previous poison survival signal,
  - optionally the previous defender action after M1/M2 works.
- Reward:
  - validation loss increase,
  - minus stealth cost,
  - optional bypass bonus.

The initial implementation should keep the attacker action 2D. Stealth belongs
in the reward first, not as a third action dimension.

### Defender

The first defender should stay in the clipped-median family and use continuous
parameters:

- Action: 2D continuous vector in `[-1, 1]`.
- Decode:
  - `clip_radius = a[0] * 2.0 + 2.5`, range `[0.5, 4.5]`.
  - `trim_ratio = (a[1] + 1) / 2 * 0.4`, range `[0, 0.4]`.
- Observation:
  - sorted update norms,
  - pairwise distance summary,
  - recent clean loss trend,
  - recent clean accuracy trend,
  - normalized round index.
- Reward:
  - clean accuracy improvement,
  - minus loss-spike penalty.

Do not add discrete defense-rule selection in the first pass. It would mix
continuous control, discrete selection, and co-learning instability at the same
time.

## Milestones

### M1: Frozen Defender, Attacker Best Response

Implement a path where the defender is fixed to a chosen clipped-median
parameter pair and the attacker is trained as a best response.

Acceptance criteria:

- A short H-step episode runs through `meta_sg` against `fl_sandbox`.
- The attacker buffer is populated from real FL transitions.
- The attacker TD3 update changes the policy without changing the defender.
- A response-curve script can sweep fixed defender clipping radii and log the
  trained attacker's mean `epsilon` and `local_steps`.

### M2: Parameterized Frozen Defender

Replace a single fixed defender with a frozen defender policy or frozen defender
parameter schedule.

Acceptance criteria:

- The same attacker BR training can run against multiple defender parameter
  settings.
- Response curves are not flat unless the experiment genuinely indicates the
  attacker cannot exploit the parameter range.
- Diagnostics report attacker reward, stealth cost, survival signal, and decoded
  defender parameters.

### M3: Single-Condition Defender Meta Update

Freeze the warm-started attacker and update the defender on one training
condition.

Acceptance criteria:

- Defender inner adaptation runs for a small number of steps.
- Reptile meta update changes defender parameters.
- Adaptation speed improves against the current attacker on the same condition.

### M4: Multi-Condition Meta Training

Sample task conditions over non-IID degree, attacker fraction, and attack budget.

Training distribution:

- `q` in `{0.1, 0.3, 0.5}`.
- attacker fraction in `{0.1, 0.2, 0.3}`.
- attack budget in `{15, 25}`.
- dataset initially MNIST.

Held-out condition:

- `q = 0.7`,
- attacker fraction `0.4`,
- attack budget `30`.

Acceptance criteria:

- Meta defender beats fixed clipped-median and single-condition defender on
  held-out adaptation speed and final clean accuracy.

### M5: Amortized Best Response

Condition the attacker policy on a compact defender-behavior embedding so small
defender changes can be handled by inference or light fine-tuning instead of a
full BR update.

Acceptance criteria:

- BR recomputation cost decreases without flattening the response curve.

## Default Development Configuration

Use small defaults until the loop is verified:

- dataset: MNIST,
- clients: 20,
- sample rate: 0.3,
- attacker fraction: 0.2,
- non-IID `q`: 0.3,
- episode horizon `H`: 30,
- benign local epochs: 1,
- benign learning rate: 0.05,
- attacker warm-start BR budget: 5 episodes per outer iteration,
- defender meta batch size: 1 condition for M3, 4 conditions for M4.

For smoke tests, use even smaller values such as `H=3`, one condition, and one
attacker update. The smoke path should prove wiring, not learning quality.

## Diagnostics And Validation

The implementation must add validation before claiming co-learning works:

1. Response curve:
   sweep defender clipping radius and plot or log attacker `epsilon` and
   `local_steps`.
2. Adaptation speed:
   measure how many FL rounds the defender needs to reach a target clean
   accuracy.
3. Held-out generalization:
   evaluate on the held-out condition and compare against fixed clipped-median
   and single-condition defender baselines.

Flat response curves are treated as a failure signal. They mean the attacker is
not actually acting as a best-response oracle for the defender.

## Implementation Boundaries

The first implementation should avoid these expansions:

- nested meta-gradient through the BR loop,
- discrete defense-rule selection,
- multiple attack families beyond the Henger-style RL attacker,
- amortized BR,
- CIFAR-10.

These can be added after M1-M4 have clear evidence.

## Open Decisions

1. Whether M1 should reuse `fl_sandbox/attacks/rl_attacker` directly or create a
   `meta_sg` wrapper around its simulator and trainer.
2. Whether defender action should stay 2D for clipped-median only or preserve
   the existing `meta_sg` 3D paper action with a disabled post-training
   dimension during early milestones.
3. Whether poison survival should be computed from clipped malicious update norm,
   cosine with aggregate update, or coordinate survival after trimmed mean.

Recommended defaults:

- create a `meta_sg` wrapper rather than changing `paper_attack.py` first,
- use 2D defender action for M1/M2 response-curve work,
- start poison survival with malicious contribution cosine and add coordinate
  survival later if needed.
