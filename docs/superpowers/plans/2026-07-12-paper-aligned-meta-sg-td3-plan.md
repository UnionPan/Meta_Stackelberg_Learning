# Paper-Aligned Meta-SG TD3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the paper-aligned Meta-SG kernel with per-round 3D Defender and 3D RL-Attacker actions, model-tail state, SB3-compatible TD3, trace-verifiable Algorithm 1 best response/leader learning and Algorithm 2 Reptile meta-learning.

**Architecture:** Build conformance bottom-up: immutable paper/scaled configuration and codecs; canonical NeuroClip-copy and model-based local-search attack; model-tail BSMG environment; role-isolated TD3; Algorithm 1; Algorithm 2; then deterministic scaled evidence and independent query gates. Keep execution, learning, Stackelberg orchestration and oracle evaluation in separate packages.

**Tech Stack:** Python 3.12, PyTorch, NumPy, dataclasses, hashlib/json, pytest; equations compatible with Stable-Baselines3 TD3 without importing legacy `meta_sg`.

---

## Task 1: Paper parameter ledger and 3D action codecs

**Files:**
- Create: `meta_stackelberg/agents/td3/__init__.py`
- Create: `meta_stackelberg/agents/td3/config.py`
- Create: `meta_stackelberg/security/defenses/paper_action.py`
- Create: `meta_stackelberg/security/attacks/rl_action.py`
- Modify: `meta_stackelberg/security/defenses/__init__.py`
- Modify: `meta_stackelberg/security/attacks/__init__.py`
- Create: `tests/meta_stackelberg/unit/test_paper_meta_sg_config.py`
- Create: `tests/meta_stackelberg/unit/test_paper_action_codecs.py`

- [ ] Write RED tests for exact paper defaults `T=100,K=10,H=(200,500),l=N_A=N_D=10`, TD3 lr/batch/gamma, FL batch/lr/client counts, immutability, serialization and paper-vs-declared provenance.
- [ ] Implement `PaperMetaSGConfig` and explicit `scaled(...)`; reject action-dimension/operator/state/update-order overrides.
- [ ] Write RED endpoint/shape/dtype/finite/round-trip tests for `PaperDefenderAction(alpha,beta,epsilon)` and `RLAttackAction(gamma,local_steps,stealth_lambda)`.
- [ ] Implement strict 3D codecs with declared bounds and integer `E` rounding; no silent clipping of invalid raw actions.
- [ ] Run focused tests and commit `feat: add paper meta-sg parameter and action contracts`.

## Task 2: Canonical post-defense and model-based RL attack operators

**Files:**
- Create: `meta_stackelberg/security/defenses/neuroclip.py`
- Create: `meta_stackelberg/security/attacks/local_search.py`
- Create: `tests/meta_stackelberg/unit/test_neuroclip.py`
- Create: `tests/meta_stackelberg/unit/test_rl_local_search_attack.py`

- [ ] Write RED tests proving NeuroClip operates on an independent model copy, is deterministic, validates epsilon, and cannot alter intermediate `RoundState`.
- [ ] Implement a versioned canonical NeuroClip transform matching the declared paper branch; keep final-delivery application separate from reward-copy evaluation.
- [ ] Write RED hand-calculated local-search tests for `(gamma,E,lambda)`, fixed `G`, shared malicious action, ascent direction, cosine term, RNG replay and capability checks.
- [ ] Implement the reference-[15] local-search attacker without importing agents/experiments/evaluation.
- [ ] Run operator/E1/E2 regressions and commit `feat: add paper defense and rl attack operators`.

## Task 3: Model-tail state, rewards and per-round BSMG environment

**Files:**
- Create: `meta_stackelberg/environments/__init__.py`
- Create: `meta_stackelberg/environments/model_tail.py`
- Create: `meta_stackelberg/environments/rewards.py`
- Create: `meta_stackelberg/environments/paper_bsmg.py`
- Create: `tests/meta_stackelberg/unit/test_model_tail_observation.py`
- Create: `tests/meta_stackelberg/unit/test_paper_rewards.py`
- Create: `tests/meta_stackelberg/integration/test_paper_bsmg_environment.py`

- [ ] Write RED tests for stable final-two-learnable-block ordering, normalizer state, malicious count/current Defender action attacker keys and no oracle/private inputs.
- [ ] Implement versioned `PaperObservation` dictionary tensors suitable for `MultiInputPolicy` semantics.
- [ ] Write RED reward-component tests using root/generated data separate from held-out query data.
- [ ] Implement immutable paper untargeted reward records.
- [ ] Write RED environment tests: one step equals one FL round, both policies act per round, action-before-update ordering, alpha/beta transition, epsilon copy-only intermediate behavior, final post-defense delivery and exact replay.
- [ ] Implement `PaperBSMGEnv` over existing canonical client/sampler/server protocols.
- [ ] Run focused/canonical tests and commit `feat: add paper aligned bsmg environment`.

## Task 4: SB3-compatible TD3, replay, snapshot and freeze contracts

**Files:**
- Create: `meta_stackelberg/agents/td3/networks.py`
- Create: `meta_stackelberg/agents/td3/replay.py`
- Create: `meta_stackelberg/agents/td3/agent.py`
- Create: `tests/meta_stackelberg/unit/test_td3_replay.py`
- Create: `tests/meta_stackelberg/unit/test_td3_agent.py`
- Create: `tests/meta_stackelberg/unit/test_td3_snapshot_freeze.py`

- [ ] Write RED replay tests for copy/freeze/capacity/batch/RNG/generation tags and dict observation flattening.
- [ ] Implement role-local replay with no cross-role transition ingestion.
- [ ] Write RED equation tests for bounded 3D actor, twin critics, `min(Q1,Q2)` target, terminal mask, target smoothing/noise clip, actor delay and Polyak tau.
- [ ] Implement canonical TD3 using declared paper/SB3 parameter provenance.
- [ ] Write RED exact snapshot/restore/fingerprint tests covering online/target/optimizers/counters/normalizer; verify opposite-role mutation is detected.
- [ ] Run deterministic CPU tests twice and commit `feat: add canonical td3 agents and freeze snapshots`.

## Task 5: Algorithm 1 trace-verifiable Stackelberg learning

**Files:**
- Create: `meta_stackelberg/stackelberg/policy_response.py`
- Create: `meta_stackelberg/stackelberg/algorithm1.py`
- Modify: `meta_stackelberg/stackelberg/__init__.py`
- Create: `tests/meta_stackelberg/unit/test_meta_sg_algorithm1_trace.py`
- Create: `tests/meta_stackelberg/integration/test_policy_level_best_response.py`

- [ ] Write pure-factory RED trace tests requiring exactly `N_D` leader iterations, `K` tasks each iteration, `N_A` attacker updates per task, `phi(N_A)` use, and no extra `N_D` inner updates.
- [ ] Implement immutable trace/response records and Algorithm 1 orchestration independent of FL details.
- [ ] Write real scaled RED integration: Defender full fingerprint unchanged throughout `N_A`; Attacker full fingerprint unchanged during Defender update; fresh post-BR trajectories only.
- [ ] Implement policy-level response trainer with persistent type-specific warm starts and independent BR query evaluation.
- [ ] Require attacker behavior/objective change, exact replay and changed response under different frozen Defender policies.
- [ ] Commit `feat: implement meta-sg algorithm1 policy responses`.

## Task 6: Algorithm 2 Reptile meta-learning

**Files:**
- Create: `meta_stackelberg/stackelberg/algorithm2.py`
- Create: `tests/meta_stackelberg/unit/test_meta_sg_algorithm2_trace.py`
- Create: `tests/meta_stackelberg/integration/test_reptile_meta_sg.py`

- [ ] Write RED trace tests for exactly `T` meta iterations, `K` sampled tasks, `l` task-adaptation updates and one Reptile meta update per `T`; reject `l/N_D` aliasing.
- [ ] Implement task clone/adapt and Reptile `theta += meta_step/K * sum(theta_xi(l)-theta)` over complete TD3 state components declared adaptable.
- [ ] Compose adaptive tasks with Algorithm 1 response refresh protocol without concurrent role updates.
- [ ] Verify meta initialization changes, task clones are isolated, attack-task sampling replay is exact and query data never selects updates.
- [ ] Commit `feat: implement reptile meta-sg algorithm2`.

## Task 7: Scaled conformance run, scientific gates and report

**Files:**
- Create: `meta_stackelberg/experiments/paper_meta_sg.py`
- Modify: `meta_stackelberg/experiments/__init__.py`
- Create: `tests/meta_stackelberg/integration/test_paper_meta_sg_scaled_run.py`
- Modify: `tests/meta_stackelberg/regression/test_dependency_boundaries.py`
- Create: `docs/milestones/E4-paper-aligned-meta-sg-report.zh-CN.md`

- [ ] Run the immutable scaled config `T=2,K=2,H=8,l=N_A=N_D=2` and verify all call-count/action/state/freeze/post-defense conformance gates.
- [ ] Add independent query comparisons: `phi(N_A)` vs `phi(0)`, two Defender policies, adapted vs initial Defender, meta vs random/no-adaptation under equal budget.
- [ ] Preserve failures rather than tuning decoder ranges/reward/query thresholds.
- [ ] Extend dependency RED/GREEN tests for agents/environment/Stackelberg/oracle boundaries.
- [ ] Report explicit paper parameters, scaled actual parameters and every deviation; never label smoke as paper reproduction.
- [ ] Run focused, canonical and full repository suites plus `git diff --check`; commit `docs: report paper aligned meta-sg scaled evidence`.

## Completion boundary

The objective is complete only when all seven tasks are implemented and verified, both Algorithm 1 and Algorithm 2 traces match their paper parameter meanings, both policies act per FL round with 3D physical semantics, role freeze and query isolation are directly evidenced, the scaled run and report exist, and fresh canonical/full suites pass. A smoke run proves conformance only; any paper-scale performance/reproduction claim requires a separately executed paper configuration.
