# fl_sandbox Attacks Restructure — Technical Design v2

This document supersedes the prior v1 spec. It keeps v1's goal of cleaning up
the attacker subsystem, but reworks the RL attacker design, dependency
boundaries, and phase split based on an RL-systems review.

---

## 1. Scope and Goals

This spec covers the cleanup of `fl_sandbox`'s attacker subsystem and the
introduction of a maintainable adaptive RL attacker. It does not refactor
defenders, aggregation runtime, the federation runner, experiment services,
postprocessing, or the `meta_sg` adapter beyond the import updates required to
keep attacker usage working.

Goals:

- Make `fl_sandbox.attacks` the single public attacker API.
- Remove attacker compatibility layers under `fl_sandbox.core`.
- Flatten fixed attackers into one file each under `fl_sandbox/attacks`.
- Move the adaptive RL attacker into a self-contained `fl_sandbox/attacks/rl_attacker/` subpackage with clear RL-side abstractions.
- Replace the hand-written TD3 implementation with **Tianshou** as the single RL backend. The hand-written `Actor` / `Critic` / `TD3Agent` / `ReplayBuffer` / target-update code is deleted, not wrapped.
- Default the RL algorithm to SAC (Tianshou `SACPolicy`); keep TD3 (Tianshou `TD3Policy`) as an alternate. Both share the same Tianshou collector and replay buffer infrastructure.
- Isolate Tianshou behind a small `Trainer` protocol so swapping algorithms (or, much later, swapping libraries) does not touch `attack.py`.
- Make sim-to-real gap a first-class, monitored quantity from day one.
- Keep each module small enough to read and test independently.

Non-goals (Phase 1):

- Defender / aggregator refactor.
- Federation runner / experiment service split.
- Broad config schema and CLI cleanup. Phase 1 may add the minimal RL
  config fields needed for Tianshou and diagnostics.
- Postprocess / reporting consolidation.
- `meta_sg` adapter alignment beyond import updates.
- Online policy training from real FL feedback (Phase 2).

---

## 2. Current Problems

Two attacker locations exist:

- `fl_sandbox/attacks` — real implementations.
- `fl_sandbox/core/attacks` and `fl_sandbox/core/rl` — re-export shims.

Within `attacks/` there is a second grouping (`vector/`, `backdoor/`,
`adaptive/`) that adds a layer without clarifying the public API.

The RL attacker file (`attacks/adaptive/td3_attacker.py`) mixes:

- proxy distribution learning,
- gradient inversion,
- FL environment simulation,
- defense-specific action decoding,
- hand-written TD3 networks, replay buffer, target updates,
- and high-level attack orchestration.

This is unmaintainable and locks the attacker to one RL algorithm. The TD3
algorithm itself is not the project's contribution and should be delegated.

---

## 3. Target Layout

### 3.1 Top-level attacker package

```text
fl_sandbox/attacks/
  __init__.py
  base.py
  registry.py

  ipm.py
  lmp.py
  alie.py
  signflip.py
  gaussian.py

  bfl.py
  dba.py
  brl.py

  krum_geometry_search.py
  clipped_median_geometry_search.py

  rl_attacker/        # see §3.2
```

Deleted paths:

```text
fl_sandbox/core/attacks/
fl_sandbox/core/rl/
fl_sandbox/attacks/vector/
fl_sandbox/attacks/backdoor/
fl_sandbox/attacks/adaptive/
```

Public import surface:

```python
from fl_sandbox.attacks import (
    ATTACK_CHOICES,
    SandboxAttack,
    create_attack,
    supported_attack_types,
    IPMAttack,
    LMPAttack,
    ALIEAttack,
    SignFlipAttack,
    GaussianAttack,
    BFLAttack,
    DBAAttack,
    BRLAttack,
    SelfGuidedBRLAttack,
    RLAttack,
    KrumGeometrySearchAttack,
    ClippedMedianGeometrySearchAttack,
)
```

### 3.2 RL attacker subpackage

```text
fl_sandbox/attacks/rl_attacker/
  __init__.py
  attack.py              # RLAttack — high-level orchestration
  config.py              # RLAttackerConfig (pure data + bounds)
  observation.py         # State compression, history window, defense encoding
  action_decoder.py      # Defense-specific action → attack-parameter decoding
  diagnostics.py         # RL + sim2real metrics

  proxy/
    __init__.py
    buffer.py            # ProxyDatasetBuffer
    learner.py           # GradientDistributionLearner, ConvDenoiser
    inversion.py         # DLG/iDLG-style inversion + quality scoring

  simulator/
    __init__.py
    env.py               # Gymnasium environment
    fl_dynamics.py       # One simulated FL round (benign + malicious + aggregate)
    reward.py            # Explicit, swappable reward function

  trainer.py             # Trainer Protocol + factory(config) -> Trainer
  tianshou_backend/
    __init__.py
    common.py            # Shared net builders, replay, collector setup
    sac.py               # SAC trainer (default)
    td3.py               # TD3 trainer (alternate)
```

### 3.3 Dependency direction

```text
attack.py
  → trainer.py (Protocol only)
  → observation.py, action_decoder.py
  → proxy/{buffer, learner, inversion}
  → simulator/env.py (only for episode setup)
  → diagnostics.py

tianshou_backend/* → trainer.py, simulator/env.py, Tianshou, torch
tianshou_backend/{sac,td3}.py → tianshou_backend/common.py
simulator/env.py → simulator/fl_dynamics.py, simulator/reward.py,
                   observation.py, action_decoder.py, Gymnasium
simulator/fl_dynamics.py → aggregators (read-only), proxy/buffer.py
proxy/* → models, torch (no cross-imports between proxy and simulator)
config.py → numpy, dataclasses only
```

Hard rules:

- `attack.py` does not import Tianshou. Importing `fl_sandbox.attacks` and
  constructing non-RL attackers must not import Tianshou or Gymnasium.
- No module under `rl_attacker/` imports from `fl_sandbox.federation.runner`
  or `fl_sandbox.core.experiment_service`.
- `simulator/` does not import from `attack.py`.

---

## 4. Module Responsibilities

### 4.1 Public entry points

- `attacks/__init__.py` — re-exports listed in §3.1. The only import path
  external code should use.
- `attacks/registry.py` — `create_attack(config)` factory and
  `ATTACK_CHOICES`. Imports concrete attacker classes.
- `attacks/base.py` — `SandboxAttack` protocol, weight capture/load helpers,
  benign-fallback local training, bounded action helpers shared across
  attackers.

### 4.2 Fixed model-poisoning attackers

One file each, depending only on `base.py`, `core.metrics`, and weight
utilities:

- `ipm.py` — `craft_ipm`, `IPMAttack`
- `lmp.py` — `craft_lmp`, `LMPAttack`
- `alie.py` — `craft_alie`, `ALIEAttack`
- `signflip.py` — `SignFlipAttack`
- `gaussian.py` — `GaussianAttack`

Cross-imports between fixed attackers are not allowed; if a helper is shared,
it lives in `base.py`.

### 4.3 Backdoor attackers

- `bfl.py` — fixed global-trigger backdoor.
- `dba.py` — distributed sub-trigger backdoor.
- `brl.py` — `BRLAttack` and `SelfGuidedBRLAttack`.

Poisoned dataset construction stays in the federation/data layer. Backdoor
attackers consume the poisoned loaders attached to `RoundContext`.

### 4.4 Geometry-search attackers

- `krum_geometry_search.py` — `KrumGeometrySearchAttack`
- `clipped_median_geometry_search.py` — `ClippedMedianGeometrySearchAttack`

Class names drop the wrapper suffix from v1. Registry keys remain
`krum_geometry_search` and `clipped_median_geometry_search`.

### 4.5 RL attacker — module contracts

#### `config.py`
Pure data. `RLAttackerConfig` carries action bounds, network sizes, training
budgets, simulator horizon, distribution-learning schedule, and references
to model/state helpers. **No defense-specific decoding lives here.**

#### `observation.py`
Builds the observation vector from a `RoundContext`:

- compressed global-model representation (random projection by default;
  config-selectable to PCA or per-layer statistics),
- per-step history window (configurable length, default 4),
- previous action,
- previous estimated defense-bypass score,
- normalized round index `t / T`,
- defense one-hot (so a single shared policy can be defense-conditioned).

Exposes `obs_space` (Gymnasium `Box`).

#### `action_decoder.py`
`decode(action, defense_type, ctx) -> AttackParameters`. One function per
defense, dispatched by string. AttackParameters is a small dataclass
consumed by `simulator/fl_dynamics.py` and by online execution in
`attack.py`. Defense-specific knowledge is centralized here, not scattered.

#### `proxy/buffer.py`
`ProxyDatasetBuffer` — accepts samples (real seed + reconstructed),
deduplicates, supports recency-weighted sampling, and tracks
`reconstruction_accept_rate`.

#### `proxy/learner.py`
Drives proxy-distribution rounds: when in distribution-learning phase,
infers the server-update gradient from successive global-model snapshots and
calls `inversion.reconstruct(...)`.

#### `proxy/inversion.py`
DLG/iDLG-style reconstruction. **Returns `(samples, quality_score)`.**
Quality scoring uses cosine similarity between the gradient produced by the
reconstructed batch and the observed gradient, plus a sanity check on
reconstructed-loss magnitude. Samples below a configured threshold are
rejected and not added to the buffer. Acceptance rate is logged.

#### `simulator/fl_dynamics.py`
Pure FL-round simulation: given current model, proxy buffer, defense config,
and a candidate `AttackParameters`, simulates benign client updates,
malicious updates, defense aggregation, and returns the new model plus
intermediate quantities the reward function needs.

#### `simulator/reward.py`
Explicit, swappable `RewardFn` interface. Default reward is a weighted sum
of:

- proxy clean-loss increase (positive),
- proxy clean-accuracy decrease (positive),
- defense-bypass indicator (e.g., malicious updates accepted by Krum),
- a small action-smoothness penalty `-||a_t - a_{t-1}||^2`,
- an out-of-bounds penalty for actions saturating the box.

Weights live in `RLAttackerConfig`. Alternate reward functions can be
registered for ablations.

#### `simulator/env.py`
Gymnasium-compatible env. `reset()` builds a fresh simulated FL trajectory
seeded from the current global model and proxy buffer. `step(action)`
decodes the action, runs `fl_dynamics`, computes reward, and returns the
next observation. `seed()` is honored for reproducibility.

Episode definition: one episode = `simulator_horizon` simulated FL rounds.
This is fixed and documented; bandit-style 1-step episodes are explicitly
not used.

#### `trainer.py`
```python
class Trainer(Protocol):
    def ensure_initialized(self, obs_space, action_space) -> None: ...
    def collect(self, env, steps: int) -> CollectStats: ...
    def update(self, gradient_steps: int) -> UpdateStats: ...
    def act(self, obs, *, deterministic: bool = False) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def diagnostics(self) -> dict: ...
```

`build_trainer(config) -> Trainer` selects a concrete trainer by
`config.algorithm` (`"sac"` default, `"td3"` alternate).

#### `tianshou_backend/`
The single RL implementation path. No alternate library is supported and no
hand-written RL primitives remain in the project.

- `common.py` — shared MLP actor/critic builders, Tianshou
  `VectorReplayBuffer` construction, `Collector` setup, action-bound
  remapping, exploration noise schedules (TD3), entropy auto-tuning hooks
  (SAC), gradient clipping, and the diagnostics-extraction helper.
- `sac.py` — `TianshouSACTrainer`. Wraps `tianshou.policy.SACPolicy` with a
  tanh-squashed Gaussian actor and twin critics. Default trainer.
- `td3.py` — `TianshouTD3Trainer`. Wraps `tianshou.policy.TD3Policy` with a
  deterministic tanh-squashed actor, twin critics, and target-policy
  smoothing.

Both trainers:

- use tanh-squashed action heads with linear remap to action bounds (no
  post-hoc clipping or NaN sanitation in the runtime path),
- enable gradient clipping (configurable, default 1.0),
- expose Q-value mean/std, TD-error, entropy (SAC), exploration noise scale
  (TD3), update step count, and replay-buffer fill via `diagnostics()`,
- support `save()` / `load()` of the full Tianshou policy state.

If Tianshou or Gymnasium is missing when `build_trainer(...)` is called,
raise a `RuntimeError` naming the missing package and the install command.
Importing `fl_sandbox.attacks.rl_attacker` (without constructing a trainer)
does **not** import Tianshou — Tianshou imports live behind the
`build_trainer` call inside `tianshou_backend/`.

#### `attack.py`
`RLAttack` (and `PaperRLAttacker` if still required by experiments) handles:

1. Seeding the proxy buffer from attacker-local loaders.
2. Driving distribution-learning rounds by calling
   `proxy.learner.observe_round(...)`.
3. Building/refreshing a `SimulatedFLEnv` when entering policy-training
   rounds.
4. Calling `trainer.collect(...)` and `trainer.update(...)` per scheduled
   simulator-step budget.
5. Online execution: build the live observation, call
   `trainer.act(obs, deterministic=True)`, decode through
   `action_decoder`, craft malicious weights via local search, and match
   norm to the defense-specific benign target.
6. `after_round(...)` records FL-side metrics and the **sim2real reward
   gap** (see §6).

#### `diagnostics.py`
Single place that publishes:

- RL training: episode return, running-mean return, critic Q stats,
  TD-error, policy entropy (SAC), action saturation rate, exploration
  noise scale, update-to-data ratio actual.
- Proxy: `reconstruction_accept_rate`, mean reconstruction quality,
  buffer size, sample age distribution.
- Sim2real: per-round `reward_real - reward_sim`, rolling mean and
  std, `policy_deploy_blocked` count.

---

## 5. Algorithm and Non-Stationarity Strategy

The FL attack environment is non-stationary: the global model evolves, the
defense's geometry shifts, and benign client distributions can drift. This
shapes three choices:

1. **Default algorithm: SAC.** Adaptive entropy gives stable exploration
   under a non-stationary target and is a better default than TD3 for this
   continuous-control setting. TD3 remains available for ablations.
2. **Recency-weighted replay.** Both trainers use exponential-recency
   weighted sampling instead of uniform FIFO. Decay rate is configurable;
   default targets a half-life of one episode-worth of transitions.
3. **Bounded buffer with high update-to-data ratio.** Default UTD = 4 with
   buffer size = 8 × episode length. This favors recent transitions over
   sheer volume.

Open-loop hyperparameter defaults (in `config.py`) are documented and
seeded; reproducibility is required.

---

## 6. Sim-to-Real Gap as a First-Class Signal

Phase 1 trains the policy entirely on the proxy simulator. The risk is the
classic exploiter-overfits-the-simulator failure: policy looks great in
sim, fails in real FL.

Phase 1 must therefore record:

- **Per-round sim2real reward gap.** When the live system executes action
  `a_t` in real FL and observes outcome `o_real`, evaluate the same `a_t`
  starting from the same state in the simulator and record
  `r_real - r_sim`. Logged through `diagnostics.py`.
- **Proxy calibration metrics.** Feature moment distance between proxy
  buffer samples and an attacker-local validation slice (kept disjoint),
  reported per distribution-learning round.
- **Deploy guard.** A new policy snapshot is only promoted to the live
  attacker if the rolling sim2real gap is within a configurable tolerance
  and the proxy calibration is above a threshold. Otherwise the previous
  snapshot is reused. Number of blocked deploys is logged.

These three are mandatory in Phase 1, even though policy training from
real-FL reward is deferred to Phase 2.

---

## 7. Runtime Flow

Per FL round (unchanged contract for the runner):

1. Benign clients train and produce benign weights.
2. Runner builds `RoundContext`.
3. `attack.observe_round(ctx)` updates RL attacker state (proxy learner,
   observation history).
4. `attack.execute(ctx)` returns malicious weights for selected attackers.
5. Defender aggregates benign + malicious updates.
6. `attack.after_round(result)` receives aggregate results, computes
   metrics, runs the sim2real evaluation, and updates diagnostics.

RL attacker internal flow:

1. Distribution-learning rounds: infer server-update gradient, reconstruct
   proxy samples, accept/reject by quality score, update proxy buffer.
2. On entering policy-training rounds, refresh `SimulatedFLEnv` from the
   current global model and proxy buffer.
3. `trainer.collect(env, simulator_steps_per_round)` then
   `trainer.update(utd * collected_steps)`.
4. Online execution: build live observation, `trainer.act(obs,
   deterministic=True)`, decode, craft malicious local-search update,
   match norm to defense-specific benign target.
5. After-round: log FL metrics, sim2real gap, proxy calibration.
6. If a policy snapshot was scheduled, run the deploy guard.

Warmup behavior: until the trainer has completed at least one update cycle,
selected attackers submit benign local updates. This preserves stable warmup
rounds and is unchanged from current behavior.

---

## 8. Observation and Action Spaces

### 8.1 Observation
Concatenation of:

- Compressed global-model vector (default: random projection to 256 dims,
  fixed projection seeded by run seed).
- History buffer of the last `H=4` (compressed model, action, reward,
  bypass score) tuples, zero-padded for early rounds.
- `t / T` round-index scalar.
- Defense one-hot.

Total dim is documented in `observation.py` and asserted at env
construction.

### 8.2 Action
Continuous `Box` of fixed dimensionality `D_a` (configured in
`RLAttackerConfig`). The policy network's final layer is **tanh-squashed**;
the env applies a linear remap to the configured bounds. NaN / Inf in
action output are treated as a critical error and trip a metric counter
(`policy_invalid_action_total`); they are not silently sanitized as a
normal code path. The previous "normalize NaN actions" rule is removed.

Defense-specific meaning of action components is fully owned by
`action_decoder.py`.

---

## 9. Error Handling

- Missing Tianshou/Gymnasium when an RL trainer is requested →
  `RuntimeError` with package name and install command.
- Importing `fl_sandbox.attacks` and constructing fixed attackers →
  must not import Tianshou or Gymnasium.
- Trainer not yet initialized at execution time → benign fallback (current
  behavior).
- Proxy reconstruction fails or scores below threshold → sample dropped,
  `reconstruction_accept_rate` updated, run continues.
- Action contains NaN/Inf → critical metric incremented, benign fallback
  for this round, no silent normalization.
- Deploy-guard rejects a snapshot → previous snapshot is kept, counter
  incremented, run continues.

---

## 10. Configuration Surface (additions)

New fields under the RL attacker config:

```text
algorithm: "sac" | "td3"      # default "sac"
simulator_horizon: int        # episode length in simulated rounds
simulator_steps_per_round: int
update_to_data_ratio: int     # default 4
replay_buffer_size: int       # default 8 * simulator_horizon
replay_recency_half_life: int # transitions; default = simulator_horizon
observation:
  compression: "random_projection" | "pca" | "layer_stats"
  compressed_dim: int          # default 256
  history_length: int          # default 4
reward:
  weights: { clean_loss, clean_acc, bypass, smoothness, oob_penalty }
deploy_guard:
  sim2real_gap_tolerance: float
  proxy_calibration_threshold: float
seed: int
```

Existing CLI flags remain available; new ones are added with sensible
defaults so existing scripts are unaffected.

---

## 11. Import Migration

Old paths to remove:

```python
from fl_sandbox.core.attacks import ...
from fl_sandbox.core.rl import ...
from fl_sandbox.attacks.vector import ...
from fl_sandbox.attacks.backdoor import ...
from fl_sandbox.attacks.adaptive import ...
```

Preferred external imports:

```python
from fl_sandbox.attacks import create_attack, ATTACK_CHOICES, RLAttack, IPMAttack
```

Internal RL package imports (tests, advanced users):

```python
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
from fl_sandbox.attacks.rl_attacker.simulator.env import SimulatedFLEnv
from fl_sandbox.attacks.rl_attacker.trainer import build_trainer
```

---

## 12. Phase Plan

The original Phase 1 bundled four risky changes. This plan splits them into
three reviewable PRs, all still inside the "Phase 1" cleanup window.

### Phase 1a — Flat layout, no RL behavior change
- Move `attacks/vector/*` and `attacks/backdoor/*` to flat files under
  `attacks/`.
- Move geometry-search attackers from `attacks/adaptive/` to flat files
  under `attacks/`.
- Move the existing RL implementation into `attacks/rl_attacker/legacy_td3.py`
  and expose it through `attacks/rl_attacker/attack.py` without changing
  algorithm behavior. This is a temporary implementation file, not a public
  compatibility layer.
- Delete `core/attacks/` and `core/rl/` shims.
- Update all in-repo imports.
- No RL algorithm changes. No new dependencies.
- Pass: existing tests + smoke `--attack_type ipm`.

### Phase 1b — RL subpackage on Tianshou (single-step replacement)
- Add Tianshou + Gymnasium to environment dependencies.
- Create `attacks/rl_attacker/` per §3.2.
- Replace the temporary `legacy_td3.py` implementation by splitting the
  adaptive attacker into `attack.py`,
  `config.py`, `observation.py`, `action_decoder.py`, `proxy/*`,
  `simulator/*`, `trainer.py`, `tianshou_backend/*`, `diagnostics.py`.
- Implement `TianshouSACTrainer` and `TianshouTD3Trainer`. Default
  algorithm is SAC.
- **Delete** the hand-written `Actor`, `Critic`, `TD3Agent`,
  `ReplayBuffer`, target-network update, and policy-noise update code in
  the same PR. There is no transitional wrapper — the project moves to
  Tianshou in one step.
- Enable the deploy guard and sim2real-gap diagnostics.
- Pass: full test set including `test_rl_attacker.py`,
  `test_attacker_sandbox_application.py`, the new unit tests in §13, and
  the Tianshou smoke command.

### Phase 2 — Online policy training (separate spec)
- Train Tianshou trainers on real-FL reward in addition to simulator.
- Keep the deploy guard but tune thresholds.
- Optional: model-based RL or CEM exploration of action space when the
  simulator is too biased.

### Later phases (as in v1)
1. Defender / aggregator boundary cleanup.
2. Federation runner / experiment service separation.
3. Config schema / CLI simplification.
4. Postprocess / reporting consolidation.
5. `meta_sg` adapter alignment.

---

## 13. Testing

Mandatory test commands per phase:

```bash
# Phase 1a
pytest tests/test_attacker_sandbox_application.py
python fl_sandbox/run/run_experiment.py \
  --attack_type ipm --rounds 1 --num_clients 4 --num_attackers 1

# Phase 1b (Tianshou installed)
pytest tests/test_rl_attacker.py tests/test_attacker_sandbox_application.py
pytest tests/rl_attacker/test_observation.py \
       tests/rl_attacker/test_action_decoder.py \
       tests/rl_attacker/test_proxy_buffer.py \
       tests/rl_attacker/test_simulator_env.py \
       tests/rl_attacker/test_tianshou_sac.py \
       tests/rl_attacker/test_tianshou_td3.py
python fl_sandbox/run/run_experiment.py \
  --attack_type rl --defense_type krum \
  --rounds 2 --num_clients 4 --num_attackers 1 \
  --rl_distribution_steps 0 \
  --rl_attack_start_round 0 \
  --rl_policy_train_end_round 1 \
  --rl_policy_train_episodes_per_round 1 \
  --rl_simulator_horizon 1
```

New test coverage to add:

- Package-level imports from `fl_sandbox.attacks`.
- `create_attack` constructs every supported attack type.
- No remaining imports from `fl_sandbox.core.attacks` or `fl_sandbox.core.rl`.
- `observation.build()` shape and determinism under seed.
- `action_decoder.decode()` for Krum, clipped median, coordinate-robust,
  FLTrust, FedAvg.
- `ProxyDatasetBuffer` recency sampling and accept/reject behavior.
- `inversion.reconstruct()` quality scoring on a synthetic gradient where
  ground truth is known.
- `SimulatedFLEnv.reset()/step()` with a tiny fake proxy buffer.
- `build_trainer({"algorithm": "sac"})` and `build_trainer({"algorithm":
  "td3"})` both initialize on the env's spaces without crashing; `act()`
  returns in-bounds actions; `diagnostics()` exposes the documented keys
  for each algorithm.
- A short `collect → update` cycle on a tiny env produces finite losses,
  finite gradients, and updated diagnostic counters for both SAC and TD3.
- `save()` then `load()` round-trips a Tianshou policy and reproduces the
  same `act()` output on a fixed observation.
- Clear `RuntimeError` when Tianshou is missing and `build_trainer` is
  called.
- No `import tianshou` occurs from `fl_sandbox.attacks` or
  `fl_sandbox.attacks.rl_attacker` at package import time (asserted by a
  test that monkeypatches `sys.modules`).
- Sim2real-gap diagnostic is emitted whenever `after_round` runs in a
  policy-training phase.

---

## 14. Documentation Updates

Update `README.md`:

- `attacks/` is the only public attacker API.
- `attacks/rl_attacker/` is the adaptive RL attacker, with SAC default and
  Tianshou backend.
- Remove references implying attackers live under `core/attacks`,
  `core/rl`, `attacks/vector`, `attacks/backdoor`, or `attacks/adaptive`.
- Document the deploy-guard and sim2real-gap diagnostics so experimenters
  know to inspect them before trusting RL-attacker results.
