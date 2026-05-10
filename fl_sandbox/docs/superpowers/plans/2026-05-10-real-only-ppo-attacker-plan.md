# Real-Only PPO Attacker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the PPO RL attacker from real FL round transitions instead of simulator rollout.

**Architecture:** TD3 keeps the existing simulator-backed training path. PPO creates/uses a Tianshou trainer against real observations, records the action actually submitted in `execute()`, and adds the real `(obs, action, reward, next_obs)` transition in `after_round()`. PPO updates when enough real transitions have accumulated, then clears its on-policy buffer.

**Tech Stack:** Python, NumPy, Gymnasium spaces, Tianshou PPO, pytest.

---

### Task 1: Trainer Real Transition API

**Files:**
- Modify: `fl_sandbox/attacks/rl_attacker/trainer.py`
- Modify: `fl_sandbox/attacks/rl_attacker/tianshou_backend/common.py`
- Test: `tests/rl_attacker/test_tianshou_trainers.py`

- [ ] Add a failing test that initializes a PPO trainer, records two real transitions with `add_transition(...)`, updates once, and verifies the PPO replay buffer is reset after update.
- [ ] Run `PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest tests/rl_attacker/test_tianshou_trainers.py::test_ppo_trainer_accepts_real_transitions_without_simulator -q`; expect failure because `add_transition` does not exist.
- [ ] Add `add_transition(obs, act, rew, obs_next, terminated=False, truncated=False)` to the trainer protocol and `BaseTianshouTrainer`, using the same `tianshou.data.Batch` shape as `collect()`.
- [ ] Re-run the targeted test; expect pass.

### Task 2: RLAttack PPO Real Rollout

**Files:**
- Modify: `fl_sandbox/attacks/rl_attacker/config.py`
- Modify: `fl_sandbox/attacks/rl_attacker/attack.py`
- Test: `tests/rl_attacker/test_real_ppo_online.py`

- [ ] Add a failing test with a fake PPO trainer proving `RLAttack.execute()` records a pending real transition and `after_round()` calls `trainer.add_transition(...)` using real clean-loss / clean-accuracy reward.
- [ ] Run the targeted test; expect failure because the pending real transition path does not exist.
- [ ] Add config field `ppo_real_rollout_steps: int = 64`.
- [ ] In `RLAttack.observe_round()`, skip simulator `_train_policy()` when `config.algorithm == "ppo"`.
- [ ] In `RLAttack.execute()`, initialize the PPO trainer from real observation/action spaces, sample stochastic actions while `round_idx <= policy_train_end_round`, record pending `(obs, action, builder_snapshot)` only for submitted PPO attack actions, and keep TD3 unchanged.
- [ ] In `RLAttack.after_round()`, build `next_obs` from the real aggregated weights, compute real reward from clean-loss/accuracy deltas plus bypass/smoothness/saturation terms, add the transition to PPO, and update once when the PPO buffer reaches `ppo_real_rollout_steps`.
- [ ] Re-run the targeted test; expect pass.

### Task 3: Verification

**Files:**
- No production changes.

- [ ] Run `PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest tests/rl_attacker tests/test_rl_attacker.py -q`; expect pass.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PATH=.venv/bin:$PATH python -m pytest -q`; expect full suite pass.
- [ ] Remove generated `__pycache__` directories under `fl_sandbox` and `tests`.
- [ ] Check `git status --short` and report only intentional files touched, leaving unrelated `meta_sg` work alone.
