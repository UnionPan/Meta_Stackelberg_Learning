# MNIST White-Box Backdoor Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable MNIST white-box `1 -> 7` Meta-SG vertical slice with the paper's three-dimensional Defender and BRL actions, real-data reward, NeuroClip, and Algorithm 1 training.

**Architecture:** Keep the existing untargeted `PaperBSMGEnv` unchanged. Add focused backdoor action, poisoning, reward, and environment modules that reuse the FL round kernel, TD3 agents, model-tail encoder, and policy-level Algorithm 1 runner. Phase A uses real MNIST data with disjoint client/reward/query roles; Prun and cGAN remain separately testable follow-on plans.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, existing TD3 and federated-learning modules.

---

## File map

- `meta_stackelberg/security/attacks/backdoor_action.py`: strict BRL `(rho, eta, E)` action and raw codec.
- `meta_stackelberg/security/data/mnist_global_trigger.py`: immutable MNIST global-trigger fixture and provenance.
- `meta_stackelberg/security/attacks/rl_backdoor.py`: per-round malicious local training with shared decoded action.
- `meta_stackelberg/environments/backdoor_rewards.py`: typed white-box Defender/Attacker reward components.
- `meta_stackelberg/environments/paper_backdoor_bsmg.py`: Defender-then-BRL sequential FL environment.
- `meta_stackelberg/experiments/paper_mnist_backdoor_env.py`: deterministic real-data splits and canonical environment factory.
- `meta_stackelberg/experiments/paper_mnist_backdoor_meta_sg.py`: TD3 dimensions, task factory, and Algorithm 1 entry point.
- `tests/meta_stackelberg/unit` and `tests/meta_stackelberg/integration`: contracts for each module.

### Task 1: BRL three-dimensional action contract

**Files:**
- Create: `meta_stackelberg/security/attacks/backdoor_action.py`
- Modify: `meta_stackelberg/security/attacks/__init__.py`
- Create: `tests/meta_stackelberg/unit/test_backdoor_action_codec.py`

- [ ] **Step 1: Write failing endpoint and validation tests**

```python
def test_backdoor_action_codec_decodes_paper_endpoints():
    codec = BackdoorActionCodec()
    assert codec.decode(np.array([-1., -1., -1.])).as_tuple() == (0.0, 0.0, 1)
    assert codec.decode(np.array([1., 1., 1.])).as_tuple() == (1.0, 0.1, 10)

@pytest.mark.parametrize('raw', [np.zeros(2), np.array([0., 0., np.nan]), np.array([0., 0., 1.1])])
def test_backdoor_action_codec_rejects_invalid_raw_actions(raw):
    with pytest.raises((ValueError, TypeError)):
        BackdoorActionCodec().decode(raw)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_backdoor_action_codec.py -q`
Expected: FAIL because `backdoor_action` does not exist.

- [ ] **Step 3: Implement the strict codec**

```python
@dataclass(frozen=True)
class BackdoorAction:
    poison_fraction: float
    learning_rate: float
    local_epochs: int

    def as_tuple(self) -> tuple[float, float, int]:
        return self.poison_fraction, self.learning_rate, self.local_epochs

class BackdoorActionCodec:
    def decode(self, raw_action: np.ndarray) -> BackdoorAction:
        raw = validate_raw_action(raw_action, dimensions=3)
        rho = round((raw[0] + 1.0) * 5.0) / 10.0
        eta = (raw[1] + 1.0) * 0.05
        epochs = min(10, max(1, round((raw[2] + 1.0) * 4.5 + 1.0)))
        return BackdoorAction(rho, eta, epochs)
```

Use a private validator equivalent to `_raw3` in `security/defenses/paper_action.py`; reject non-floating arrays, non-finite values, wrong shape, and values outside `[-1, 1]`.

- [ ] **Step 4: Verify GREEN and regression scope**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_backdoor_action_codec.py tests/meta_stackelberg/unit/test_paper_action_codecs.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/security/attacks/backdoor_action.py meta_stackelberg/security/attacks/__init__.py tests/meta_stackelberg/unit/test_backdoor_action_codec.py
git commit -m "feat: add paper BRL backdoor action codec"
```

### Task 2: Immutable MNIST global-trigger task

**Files:**
- Create: `meta_stackelberg/security/data/mnist_global_trigger.py`
- Modify: `meta_stackelberg/security/data/__init__.py`
- Create: `tests/meta_stackelberg/unit/test_mnist_global_trigger.py`

- [ ] **Step 1: Write failing fixture tests**

```python
def test_mnist_global_trigger_is_immutable_and_hash_stable():
    fixture = mnist_global_trigger()
    image = torch.zeros(1, 28, 28)
    triggered = fixture.trigger.apply(image)
    assert fixture.source_class == 1
    assert fixture.target_class == 7
    assert fixture.identifier == 'mnist-global-1-to-7-v1'
    assert fixture.sha256 == fixture.compute_sha256()
    assert torch.equal(image, torch.zeros_like(image))
    assert int(torch.count_nonzero(triggered)) == len(fixture.pixels)
    assert fixture.pixels == tuple(
        (row, column, 2.82148653034729)
        for row in range(5, 7) for column in range(6, 11)
    )
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_mnist_global_trigger.py -q`
Expected: FAIL because the fixture module is absent.

- [ ] **Step 3: Implement the checked fixture**

```python
@dataclass(frozen=True)
class MNISTGlobalTriggerFixture:
    identifier: str
    source_class: int
    target_class: int
    pixels: tuple[tuple[int, int, float], ...]
    sha256: str

    @property
    def trigger(self) -> CompositeTrigger:
        return CompositeTrigger(tuple(PatchTrigger(r, c, 1, 1, v) for r, c, v in self.pixels))

    def compute_sha256(self) -> str:
        payload = json.dumps(self.pixels, separators=(',', ':')).encode()
        return hashlib.sha256(payload).hexdigest()
```

Use rows `5:7`, columns `6:11`, and normalized-white value
`(1.0 - 0.1307) / 0.3081 == 2.82148653034729`, matching the pinned raw-pixel value `255` before MNIST
`ToTensor` and normalization. Store the literal hash
`c5226726b8b70efa2f667d59784a81e6f3f7e1a359a53d067b2db086d570c93b`. Do not infer a new geometry.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_mnist_global_trigger.py tests/meta_stackelberg/unit/test_backdoor_data.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/security/data/mnist_global_trigger.py meta_stackelberg/security/data/__init__.py tests/meta_stackelberg/unit/test_mnist_global_trigger.py
git commit -m "feat: add MNIST global backdoor fixture"
```

### Task 3: RL backdoor round generator

**Files:**
- Create: `meta_stackelberg/security/attacks/rl_backdoor.py`
- Create: `tests/meta_stackelberg/unit/test_rl_backdoor_attack.py`

- [ ] **Step 1: Write failing determinism and shared-action tests**

```python
def test_rl_backdoor_uses_one_action_for_all_malicious_clients(backdoor_round_fixture):
    attack = RLBackdoorAttack(action=BackdoorAction(0.5, 0.05, 3), **backdoor_round_fixture.kwargs)
    first = attack.craft_round(backdoor_round_fixture.context, backdoor_round_fixture.rngs())
    second = attack.craft_round(backdoor_round_fixture.context, backdoor_round_fixture.rngs())
    assert tuple(update.client_id for update in first) == backdoor_round_fixture.context.malicious_client_ids
    assert [update.metadata['decoded_action'] for update in first] == [(0.5, 0.05, 3)] * len(first)
    assert [update.delta for update in first] == [update.delta for update in second]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_rl_backdoor_attack.py -q`
Expected: FAIL because `RLBackdoorAttack` is missing.

- [ ] **Step 3: Implement trainer construction per malicious client**

```python
class RLBackdoorAttack:
    def craft_round(self, context: RoundAttackContext, rngs: tuple[RandomSource, ...]) -> tuple[ClientUpdate, ...]:
        updates = []
        for client_id, rng in zip(context.malicious_client_ids, rngs, strict=True):
            poisoned = SourceTargetPoisonedDataset(
                self.client_datasets[client_id], self.trigger,
                source_class=1, target_class=7,
                poison_fraction=self.action.poison_fraction,
                seed=rng.seed,
            )
            trainer = TorchLocalTrainer(
                model_factory=self.model_factory,
                client_datasets={client_id: poisoned}, codec=self.codec,
                learning_rate=self.action.learning_rate,
                local_epochs=self.action.local_epochs, batch_size=self.batch_size,
            )
            updates.append(mark_backdoor(trainer.train(client_id, make_state(context, rng), rng), self.action))
        return tuple(updates)
```

Use `TorchLocalTrainer.train(client_id, ClientState(global_model, rng.capture()), rng)` and construct
`ClientUpdate` metadata with `attack_type`, `source_class`, `target_class`, and `decoded_action` keys.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_rl_backdoor_attack.py tests/meta_stackelberg/integration/test_backdoor_attack_validity.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/security/attacks/rl_backdoor.py tests/meta_stackelberg/unit/test_rl_backdoor_attack.py
git commit -m "feat: add RL backdoor round attack"
```

### Task 4: White-box joint rewards

**Files:**
- Create: `meta_stackelberg/environments/backdoor_rewards.py`
- Create: `tests/meta_stackelberg/unit/test_backdoor_rewards.py`

- [ ] **Step 1: Write failing component and sign tests**

```python
def test_whitebox_rewards_keep_defender_and_attacker_objectives_distinct():
    defender, attacker = evaluate_whitebox_backdoor_rewards(
        clean_loss=0.2, safe_loss=0.4, target_loss=0.1,
        clean_damage=0.05, defender_lambda=0.5, attacker_lambda=0.5,
    )
    assert defender.scalar == pytest.approx(-0.3)
    assert attacker.scalar == pytest.approx(-0.075)
    assert defender.source == 'mnist-whitebox-real-data-v1'
    assert attacker.target_loss == 0.1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_backdoor_rewards.py -q`
Expected: FAIL because reward types are missing.

- [ ] **Step 3: Implement immutable reward records and validation**

```python
def evaluate_whitebox_backdoor_rewards(*, clean_loss, safe_loss, target_loss,
                                       clean_damage, defender_lambda, attacker_lambda):
    d = -((1.0 - defender_lambda) * clean_loss + defender_lambda * safe_loss)
    a = -((1.0 - attacker_lambda) * target_loss + attacker_lambda * clean_damage)
    return WhiteBoxDefenderReward(d, clean_loss, safe_loss), WhiteBoxAttackerReward(
        a, target_loss, clean_damage,
    )
```

Validate all losses as finite and non-negative, and both lambdas within `[0, 1]`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/meta_stackelberg/unit/test_backdoor_rewards.py tests/meta_stackelberg/unit/test_paper_rewards.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/environments/backdoor_rewards.py tests/meta_stackelberg/unit/test_backdoor_rewards.py
git commit -m "feat: add white-box backdoor rewards"
```

### Task 5: Sequential backdoor BSMG environment

**Files:**
- Create: `meta_stackelberg/environments/paper_backdoor_bsmg.py`
- Create: `tests/meta_stackelberg/integration/test_paper_backdoor_bsmg_environment.py`

- [ ] **Step 1: Write failing phase-order and action-effect tests**

```python
def test_backdoor_bsmg_executes_defender_then_shared_brl_action(env):
    pending = env.begin_round(np.zeros(3, dtype=np.float64))
    assert pending.attacker_observation['defender_raw_action'].shape == (3,)
    step = env.finish_round(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert step.attacker_action == BackdoorAction(0.5, 0.05, 6)
    assert step.transition.state_after.round_index == 1
    assert step.defender_reward.source == 'mnist-whitebox-real-data-v1'
    assert step.transition.private_diagnostics['malicious_client_count'] >= 0
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_paper_backdoor_bsmg_environment.py -q`
Expected: FAIL because the environment is missing.

- [ ] **Step 3: Implement by extracting shared round mechanics, not branching untargeted behavior**

```python
class PaperBackdoorBSMGEnv:
    def begin_round(self, defender_raw_action):
        raw = validate_raw_action(defender_raw_action)
        action = self.defender_codec.decode(raw, observed_max_norm=self.observed_max_norm)
        pending = prepare_backdoor_round(self, raw, action)
        self._pending = pending
        return pending.public

    def finish_round(self, attacker_raw_action):
        action = self.attacker_codec.decode(attacker_raw_action)
        malicious = self.attack_factory(action).craft_round(
            self._pending.attack_context, self._pending.malicious_rngs,
        )
        ordered_updates = self._pending.merge_updates(malicious)
        transition = finalize_round(
            request=self._pending.request, parent_rng=self.rng,
            sampled_clients=self._pending.sampled_clients,
            ordered_updates=ordered_updates,
            aggregator=ClippedTrimmedMean(
                self._pending.defender_action.alpha,
                self._pending.defender_action.beta,
            ),
            server_optimizer=self.server_optimizer,
            private_diagnostics=self._pending.diagnostics,
        )
        defended_model = self.post_defense_copy(
            transition.state_after, self._pending.defender_action.epsilon,
        )
        rewards = self.reward_evaluator.evaluate(defended_model)
        return make_backdoor_step(self._pending, transition, action, rewards)
```

Define `prepare_backdoor_round`, `merge_updates`, and `make_backdoor_step` in the same module as private,
fully typed helpers using the slot ordering from `PaperBSMGEnv`. Do not modify `PaperBSMGEnv` behavior. The
environment constructor accepts separate reward datasets and never receives held-out query data.

- [ ] **Step 4: Verify GREEN plus untargeted non-regression**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_paper_backdoor_bsmg_environment.py tests/meta_stackelberg/integration/test_paper_bsmg_environment.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/environments/paper_backdoor_bsmg.py tests/meta_stackelberg/integration/test_paper_backdoor_bsmg_environment.py
git commit -m "feat: add white-box backdoor BSMG environment"
```

### Task 6: Canonical real-data MNIST factory and split isolation

**Files:**
- Create: `meta_stackelberg/experiments/paper_mnist_backdoor_env.py`
- Create: `tests/meta_stackelberg/integration/test_paper_mnist_backdoor_environment.py`

- [ ] **Step 1: Write failing split and paper-configuration tests**

```python
def test_whitebox_factory_keeps_full_client_train_and_isolates_query(tiny_mnist):
    bundle = split_whitebox_mnist(tiny_mnist, seed=17, reward_samples=200)
    assert len(bundle.client_train) == 60_000
    assert set(bundle.reward_indices).issubset(set(range(60_000)))
    assert bundle.query is not bundle.client_train
    factory = PaperMNISTBackdoorEnvironmentFactory.from_bundle(bundle, seed=17)
    assert factory.workers == 100
    assert len(factory.malicious_ids) == 5
    assert sum(len(part.indices) for part in factory.client_datasets.values()) == 60_000
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_paper_mnist_backdoor_environment.py -q`
Expected: FAIL because the factory is absent.

- [ ] **Step 3: Implement typed bundle and factory**

```python
@dataclass(frozen=True)
class WhiteBoxMNISTBundle:
    client_train: Dataset
    reward: Dataset
    query: Dataset
    reward_indices: tuple[int, ...]

class PaperMNISTBackdoorEnvironmentFactory:
    workers = 100
    sample_size = 10
    poison_fraction = 0.5

    def make(self, *, seed: int, horizon: int) -> PaperBackdoorBSMGEnv:
        source = RandomSource(seed)
        return PaperBackdoorBSMGEnv(
            task_id='mnist-whitebox-real-data-v1',
            initial_state=RoundState(0, self.initial_global_model, source.capture()),
            rng=source, horizon=horizon, sample_size=self.sample_size,
            sampler=UniformClientSampler(self.workers),
            benign_trainer=self.benign_trainer,
            population=FixedMaliciousPopulation(self.malicious_ids),
            model_factory=self.model_factory, codec=self.codec,
            malicious_client_datasets=self.client_datasets,
            reward_dataset=self.bundle.reward,
            observation_encoder=self.observation_encoder,
            server_optimizer=ServerSGD(), trigger=self.trigger,
        )
```

The initializer builds paper-q partitions, the first five deterministic malicious IDs, `PaperMNISTCNN`,
`ModelTailObservationEncoder`, `TorchParameterCodec`, batch 128, benign local epoch 1, and learning rate 0.05.
Validate index identity rather than only dataset lengths.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_paper_mnist_backdoor_environment.py tests/meta_stackelberg/integration/test_paper_mnist_environment.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/experiments/paper_mnist_backdoor_env.py tests/meta_stackelberg/integration/test_paper_mnist_backdoor_environment.py
git commit -m "feat: build canonical MNIST white-box environment"
```

### Task 7: Algorithm 1 TD3 wiring and smoke evidence

**Files:**
- Create: `meta_stackelberg/experiments/paper_mnist_backdoor_meta_sg.py`
- Create: `tests/meta_stackelberg/integration/test_mnist_backdoor_meta_sg.py`
- Modify: `meta_stackelberg/experiments/run_paper_evidence.py`
- Modify: `tests/meta_stackelberg/unit/test_paper_evidence_cli.py`

- [ ] **Step 1: Write failing policy-dimension and nesting tests**

```python
def test_mnist_backdoor_runner_uses_two_3d_policies_and_adapted_defender(tmp_path, bundle):
    result = run_mnist_whitebox_backdoor_meta_sg(
        bundle=bundle, output_dir=tmp_path, seed=9,
        config=smoke_config(k=2, n_a=2, n_d=1, horizon=2),
    )
    assert result.defender_action_dim == 3
    assert result.attacker_action_dim == 3
    assert result.trace[0].attacker_response.defender_origin == 'adapted'
    assert result.trace[0].attacker_response.update_steps == 2
    assert result.manifest['protocol'] == 'mnist-whitebox-real-data-v1'
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_mnist_backdoor_meta_sg.py -q`
Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement the runner using existing policy Algorithm 1**

```python
def run_mnist_whitebox_backdoor_meta_sg(*, bundle, output_dir, seed, config):
    env_factory = PaperMNISTBackdoorEnvironmentFactory.from_bundle(bundle, seed=seed)
    observation_dim = env_factory.observation_encoder.defender_dimension
    defender = make_paper_td3_agent(observation_dim, 3, config, seed)
    attacker = TD3Agent(TD3Config(
        observation_dim=env_factory.attacker_observation_dim,
        action_dim=3,
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.policy_learning_rate,
        batch_size=config.td3_batch_size,
        gamma=config.gamma,
        tau=config.tau,
        replay_capacity=config.replay_capacity,
        learning_starts=config.learning_starts,
        seed=seed + 1,
    ))
    result = run_policy_meta_sg_algorithm1(
        defender=defender, attack_domain=make_backdoor_attack_domain(attacker),
        environment_factory=env_factory, config=config,
    )
    write_atomic_manifest(output_dir, result, protocol='mnist-whitebox-real-data-v1')
    return result
```

Add `make_paper_td3_agent` as a private constructor using the same complete `TD3Config` fields shown for the
Attacker. Adapt only field spelling where the existing dataclass differs. Preserve paper defaults in the public
config: `K=10`, `N_A=10`, `N_D=10`, `H=200`, `eta=0.01`, and both adaptation rates `0.001`; smoke tests
explicitly override budgets.

- [ ] **Step 4: Add CLI mode and verify GREEN**

Run: `.venv/bin/pytest tests/meta_stackelberg/integration/test_mnist_backdoor_meta_sg.py tests/meta_stackelberg/unit/test_paper_evidence_cli.py tests/meta_stackelberg/integration/test_policy_meta_sg_algorithm1.py -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meta_stackelberg/experiments/paper_mnist_backdoor_meta_sg.py meta_stackelberg/experiments/run_paper_evidence.py tests/meta_stackelberg/integration/test_mnist_backdoor_meta_sg.py tests/meta_stackelberg/unit/test_paper_evidence_cli.py
git commit -m "feat: run MNIST white-box backdoor Meta-SG"
```

### Task 8: Phase A verification and documentation

**Files:**
- Create: `docs/milestones/mnist-whitebox-backdoor-phase-a.md`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
.venv/bin/pytest \
  tests/meta_stackelberg/unit/test_backdoor_action_codec.py \
  tests/meta_stackelberg/unit/test_mnist_global_trigger.py \
  tests/meta_stackelberg/unit/test_rl_backdoor_attack.py \
  tests/meta_stackelberg/unit/test_backdoor_rewards.py \
  tests/meta_stackelberg/integration/test_paper_backdoor_bsmg_environment.py \
  tests/meta_stackelberg/integration/test_paper_mnist_backdoor_environment.py \
  tests/meta_stackelberg/integration/test_mnist_backdoor_meta_sg.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run the full regression suite**

Run: `.venv/bin/pytest -q`
Expected: all tests PASS with no new warnings or failures.

- [ ] **Step 3: Execute deterministic smoke run twice**

Run the new CLI with `K=2`, `N_A=2`, `N_D=1`, `H=2`, seed `1701`, to two output directories.
Expected: both commands exit 0; manifests contain identical action traces, task hashes, rewards, and final policy fingerprints.

- [ ] **Step 4: Record evidence without claiming scientific success**

```markdown
# MNIST White-Box Backdoor Phase A

- Protocol: `mnist-whitebox-real-data-v1`
- Trigger task: source 1, target 7, fixture hash copied verbatim from `manifest.json`
- Defender action: `(alpha, beta, epsilon)`
- Attacker action: `(rho, eta, E)`
- Result: execution/determinism smoke evidence only
- Scientific Gate: not evaluated by the scaled smoke run
```

- [ ] **Step 5: Commit**

```bash
git add docs/milestones/mnist-whitebox-backdoor-phase-a.md
git commit -m "docs: record MNIST white-box phase A evidence"
```

## Follow-on plans

After Phase A is green, write two separate plans:

1. `mnist-prun-defense`: activation-based Prun implementation, `(alpha,beta,sigma)` codec, and matched NeuroClip/Prun evidence.
2. `mnist-generated-data`: original 200-sample seed, 100-epoch cGAN without augmentation, 60,000 generated
   examples, provenance, and generated-data comparison. A separate later plan covers the matched 5,000-example
   paper-augmentation reproduction.
