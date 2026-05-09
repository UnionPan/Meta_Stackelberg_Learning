# fl_sandbox RL Attacker Tianshou Phase 1b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `rl_attacker/legacy_td3.py` with the Tianshou-backed RL attacker package described in the restructure spec.

**Architecture:** Split the RL attacker into config, observation, action decoding, proxy learning, simulator, trainer protocol, and Tianshou backend modules. `RLAttack` uses the trainer protocol and defaults to SAC. No hand-written `Actor`, `Critic`, `TD3Agent`, or `ReplayBuffer` remains.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Gymnasium, Tianshou 2.x, pytest.

---

## Tasks

### Task 1: Dependencies and Tests

- Add `fl_sandbox/envs/requirements.txt` with Tianshou and Gymnasium.
- Add Phase 1b tests under `tests/rl_attacker/`.
- Update existing RL tests to import new modules.

### Task 2: Split RL Modules

- Create `config.py`, `action_decoder.py`, `observation.py`, `diagnostics.py`.
- Move proxy buffer/learner/inversion logic into `proxy/`.
- Move simulator dynamics/reward/env into `simulator/`.
- Preserve `GradientDistributionLearner`, `ProxyDatasetBuffer`, `SimulatedFLEnv`, and `local_search_update` public behavior through the new modules.

### Task 3: Tianshou Trainers

- Add `trainer.py` with the `Trainer` protocol and `build_trainer`.
- Add `tianshou_backend/common.py`, `sac.py`, and `td3.py`.
- Default algorithm is `sac`; `td3` is selectable.
- Trainer construction imports Tianshou lazily.

### Task 4: Replace Runtime Wiring

- Update `attack.py`, `__init__.py`, `registry.py`, `brl.py`, and tests.
- Delete `legacy_td3.py`, old `env.py`, and old `pz_env.py`.
- Ensure no hand-written TD3 primitive names remain in source.

### Task 5: Verification

- Run the full test suite.
- Run the Phase 1b Tianshou smoke command.
- Verify `fl_sandbox.attacks` import does not import Tianshou.

