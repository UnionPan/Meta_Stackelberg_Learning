# fl_sandbox Attacks Restructure Design

## Scope

This spec covers Phase 1 of the `fl_sandbox` cleanup: restructure the attacker subsystem so it is clear, direct, and ready for adaptive RL attackers. It does not refactor defenders, aggregation runtime, experiment services, postprocessing, or the `meta_sg` adapter beyond import updates needed to keep attacker usage working.

The main goals are:

- Make `fl_sandbox.attacks` the only public attacker API.
- Remove attacker compatibility layers under `fl_sandbox.core`.
- Split fixed attackers into direct files under `fl_sandbox/attacks`.
- Move the adaptive RL attacker into `fl_sandbox/attacks/rl_attacker`.
- Replace the hand-written TD3 implementation with Tianshou.
- Keep each module small enough to understand and test independently.

## Current Problems

The current tree has two attacker locations:

- `fl_sandbox/attacks`, which contains the real implementations.
- `fl_sandbox/core/attacks` and `fl_sandbox/core/rl`, which mostly re-export the new implementations as compatibility shims.

There is also an older implementation grouping inside `attacks/vector`, `attacks/backdoor`, and `attacks/adaptive`. That grouping helped during the initial sandbox build, but it makes the public API unclear and leaves multiple import styles in use.

The RL attacker is also too monolithic. `attacks/adaptive/td3_attacker.py` mixes distribution learning, gradient inversion, environment simulation, action decoding, hand-written TD3 networks, replay buffer logic, and high-level attack orchestration. The TD3 algorithm code is not the domain-specific part of this project and should be delegated to a maintained RL library.

## Target Layout

The attacker package should become:

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

  rl_attacker/
    __init__.py
    attack.py
    config.py
    distribution.py
    simulator.py
    env.py
    tianshou_policy.py
```

Delete these old implementation or compatibility paths:

```text
fl_sandbox/core/attacks/
fl_sandbox/core/rl/
fl_sandbox/attacks/vector/
fl_sandbox/attacks/backdoor/
fl_sandbox/attacks/adaptive/
```

The package-level public API is:

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

Project code and external scripts should import attackers from `fl_sandbox.attacks`. Submodules remain normal Python modules, but they are implementation details unless a test needs a specific internal helper.

## Module Responsibilities

### Public Entry Points

`attacks/__init__.py` is the single public import surface. It re-exports attacker classes, helper crafting functions, `ATTACK_CHOICES`, `create_attack`, and `supported_attack_types`.

`attacks/registry.py` builds attackers from config. It imports from the new flat modules and from `attacks.rl_attacker`. It remains available through `fl_sandbox.attacks.create_attack`, not as the preferred external import path.

`attacks/base.py` keeps shared attacker protocol code: `SandboxAttack`, weight capture/loading helpers, local benign fallback training, and bounded action helpers.

### Fixed Model-Poisoning Attackers

Each fixed model-poisoning attacker gets one file:

- `ipm.py`: `craft_ipm`, `IPMAttack`
- `lmp.py`: `craft_lmp`, `LMPAttack`
- `alie.py`: `craft_alie`, `ALIEAttack`
- `signflip.py`: `SignFlipAttack`
- `gaussian.py`: `GaussianAttack`

These files depend on `attacks.base`, `core.metrics`, and existing weight utilities only where needed. They should not import from other attacker files unless there is a real shared helper that belongs in `base.py`.

### Backdoor Attackers

Backdoor attacks move out of `attacks/backdoor/__init__.py`:

- `bfl.py`: fixed global-trigger backdoor attacker.
- `dba.py`: distributed sub-trigger backdoor attacker.
- `brl.py`: BRL and self-guided BRL variants.

The poisoned dataset construction stays in the federation/data layer. Backdoor attacker files consume the poisoned loaders already attached to `RoundContext`.

### Geometry Search Attackers

Defense-specific geometry search attackers move directly under `attacks/`:

- `krum_geometry_search.py`
- `clipped_median_geometry_search.py`

Their public class names should drop the wrapper suffix:

```python
KrumGeometrySearchAttack
ClippedMedianGeometrySearchAttack
```

The registry may still map attack types named `krum_geometry_search` and `clipped_median_geometry_search`.

### RL Attacker Package

The adaptive RL attacker is complex enough to be a subpackage:

```text
attacks/rl_attacker/
  __init__.py
  attack.py
  config.py
  distribution.py
  simulator.py
  env.py
  tianshou_policy.py
```

`attack.py` contains `RLAttack` and `PaperRLAttacker`. It is the high-level orchestration layer. It observes each FL round, updates the proxy distribution, trains the policy when scheduled, chooses an action, decodes it, crafts malicious weights, and returns one malicious update per selected attacker.

`config.py` contains `RLAttackerConfig` and `DecodedAction`. It owns action bounds, action decoding, defense-specific strategy selection, norm target ratios, and state flattening.

`distribution.py` contains `ProxyDatasetBuffer`, `GradientDistributionLearner`, `ConvDenoiser`, and gradient inversion helpers. It learns a proxy benign-data distribution from attacker-local seed data and observed server updates.

`simulator.py` contains `SimulatedFLEnv`, local malicious search, proxy benign update simulation, proxy metric evaluation, and defense-bypass reward helpers. It owns FL-domain simulation, not RL algorithm internals.

`env.py` exposes a Gymnasium-compatible environment around `SimulatedFLEnv`. Its observation space is the compressed model state plus selected attacker count. Its action space is the continuous action box from `RLAttackerConfig`.

`tianshou_policy.py` owns the Tianshou integration. It creates the actor, critic, `TD3Policy`, replay buffer, collector, and training loop adapter. It exposes a small project-facing API, for example:

```python
trainer = TianshouTD3Trainer(config, device)
trainer.ensure_initialized(obs_space, action_space)
trainer.train(env, episodes=n)
action = trainer.act(state)
```

The project should no longer maintain custom `Actor`, `Critic`, `TD3Agent`, `ReplayBuffer`, target-network update logic, or TD3 policy-noise updates.

## Tianshou Dependency

Add Tianshou and Gymnasium to the sandbox environment dependencies. PettingZoo is not required for this single-attacker environment and should be removed from the RL attacker path unless a later multi-agent attacker design needs it.

The intended dependency direction is:

```text
attack.py -> config.py, distribution.py, simulator.py, tianshou_policy.py
tianshou_policy.py -> env.py, config.py, Tianshou
env.py -> simulator.py, config.py, Gymnasium
simulator.py -> config.py, distribution.py, aggregators
distribution.py -> models, torch
config.py -> models/state helpers, numpy
```

No lower-level RL attacker module should import from `attack.py`.

## Runtime Flow

The FL runner flow remains the same:

1. Benign clients train and produce benign weights.
2. The runner builds `RoundContext`.
3. `attack.observe_round(ctx)` updates RL attacker state.
4. `attack.execute(ctx)` returns malicious weights for selected attackers.
5. The defender aggregates benign plus malicious weights.
6. If the attacker implements feedback, `attack.after_round(...)` receives aggregate results and returns attack metrics.

The RL attacker flow is:

1. Seed the proxy buffer from attacker loaders.
2. During distribution-learning rounds, infer server update gradients from global model differences and reconstruct proxy samples.
3. Build a simulated FL environment from the proxy buffer, current model, defense config, and FL config.
4. Train or update a Tianshou TD3 policy in the simulator.
5. Execute the learned action online by crafting malicious local-search updates.
6. Match malicious update norm to the defense-specific benign norm target.
7. Implement `after_round` for RL attack metrics, including real clean-loss delta, clean-accuracy delta, malicious update norm, and defense-bypass diagnostics where available. Phase 1 records these metrics but does not train Tianshou from real FL feedback; that can be a later adaptive-feedback phase.

## Import Migration

All project imports should move away from old paths:

```python
from fl_sandbox.core.attacks import ...
from fl_sandbox.core.rl import ...
from fl_sandbox.attacks.vector import ...
from fl_sandbox.attacks.backdoor import ...
from fl_sandbox.attacks.adaptive import ...
```

Preferred public imports:

```python
from fl_sandbox.attacks import create_attack, ATTACK_CHOICES, RLAttack, IPMAttack
```

Internal RL package imports may use:

```python
from fl_sandbox.attacks.rl_attacker.config import RLAttackerConfig
```

Tests may import internal helpers from `rl_attacker` when they specifically test distribution learning, simulation, or Tianshou policy setup.

## Error Handling

If Tianshou or Gymnasium is missing and RL policy training is requested, raise an actionable `RuntimeError` explaining which dependency is missing and how to install it. Importing `fl_sandbox.attacks` and constructing non-RL attackers must not import Tianshou.

If a policy is not ready, `RLAttack.execute` keeps the current fallback behavior: selected attackers submit benign local updates or unchanged weights. This keeps warmup rounds stable.

If proxy reconstruction fails in a round, the attacker logs no hard failure and continues with the existing proxy buffer. A single bad inversion should not abort a long FL run.

If Tianshou action output contains NaN or infinite values, normalize with the existing action sanitation rules in `RLAttackerConfig` before decoding.

## Testing

The minimum test pass for this phase is:

```bash
pytest tests/test_rl_attacker.py tests/test_attacker_sandbox_application.py
```

Add or update tests for:

- Package-level imports from `fl_sandbox.attacks`.
- `create_attack` constructing every supported attack type.
- No remaining imports from `fl_sandbox.core.attacks` or `fl_sandbox.core.rl`.
- RL config action decoding for Krum, clipped median, coordinate robust defenses, FLTrust, and FedAvg.
- RL proxy distribution buffer and gradient observation behavior.
- RL simulator reset/step with a small fake proxy buffer.
- Tianshou trainer initialization when dependencies are installed.
- Clear error message when Tianshou is unavailable and `RLAttack` training is requested.

Run a light smoke experiment after tests:

```bash
python fl_sandbox/run/run_experiment.py \
  --attack_type ipm \
  --rounds 1 \
  --num_clients 4 \
  --num_attackers 1
```

If Tianshou is installed, also run a tiny RL smoke test with very small budgets:

```bash
python fl_sandbox/run/run_experiment.py \
  --attack_type rl \
  --defense_type krum \
  --rounds 2 \
  --num_clients 4 \
  --num_attackers 1 \
  --rl_distribution_steps 0 \
  --rl_attack_start_round 0 \
  --rl_policy_train_end_round 1 \
  --rl_policy_train_episodes_per_round 1 \
  --rl_simulator_horizon 1
```

## Documentation Updates

Update `README.md` so the folder layout describes:

- `attacks/` as the attacker subsystem and only public attacker API.
- `attacks/rl_attacker/` as the adaptive RL attacker implementation.
- Tianshou as the RL training backend.

Remove references that imply attacker implementations live under `core/attacks`, `core/rl`, `attacks/vector`, `attacks/backdoor`, or `attacks/adaptive`.

## Out of Scope for Phase 1

Do not refactor these areas in this phase except for import updates required by the attacker API:

- `core/defender`
- `aggregators`
- `federation/runner`
- `core/experiment_service`
- `config/schema`
- `core/postprocess`
- `application`
- `meta_sg`

Those should be handled in later phases.

## Later fl_sandbox Cleanup Phases

After Phase 1, the broader `fl_sandbox` cleanup can proceed in smaller specs:

1. Defender and aggregator boundary cleanup.
2. Federation runner and experiment service separation.
3. Config schema and CLI simplification.
4. Postprocess/reporting consolidation.
5. `meta_sg` adapter alignment with the final public sandbox APIs.

This order keeps each change testable and avoids mixing attacker behavior changes with unrelated infrastructure movement.
