# E0 Core And Deterministic Clean Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first canonical `meta_stackelberg` vertical slice: immutable model/update records, explicit reproducible RNG, plug-in FL protocols, sample-weighted FedAvg, server update, and a deterministic synthetic clean round.

**Architecture:** Canonical code does not import legacy packages. A `RoundEngine` orchestrates injected `ClientSampler`, `LocalTrainer`, `Aggregator`, and `ServerOptimizer` implementations using typed records. The first slice uses scripted test plugins rather than datasets or neural networks so update semantics, data flow, reproducibility, and plugability can be proven independently.

**Tech Stack:** Python 3.12 in the current environment, NumPy, optional PyTorch only inside `RandomSource`, pytest.

---

## Task 1: Canonical Package And Dependency Guard

**Files:**
- Create: `meta_stackelberg/__init__.py`
- Create: `meta_stackelberg/core/__init__.py`
- Create: `meta_stackelberg/federated/__init__.py`
- Create: `meta_stackelberg/federated/aggregation/__init__.py`
- Create: `meta_stackelberg/federated/engine/__init__.py`
- Create: `tests/meta_stackelberg/regression/test_dependency_boundaries.py`

- [ ] **Step 1: Write the failing dependency-boundary test**

The test parses Python imports under `meta_stackelberg/` and rejects direct imports from `fl_sandbox`, `meta_sg`, and `src`.

```python
def test_canonical_package_does_not_import_legacy_packages():
    violations = find_imports(('fl_sandbox', 'meta_sg', 'src'))
    assert violations == []
```

- [ ] **Step 2: Run the test and verify package discovery fails**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/regression/test_dependency_boundaries.py -q`

Expected: FAIL because the canonical package does not exist.

- [ ] **Step 3: Create lightweight package files**

`meta_stackelberg/__init__.py` contains `__version__ = '0.1.0'`. Subpackage initializers contain docstrings only and do not import NumPy or Torch.

- [ ] **Step 4: Verify import and boundary tests**

Run:

```bash
/Users/antik/anaconda3/bin/python3 -c "import meta_stackelberg"
/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/regression/test_dependency_boundaries.py -q
```

Expected: PASS.

## Task 2: Immutable ModelState And Delta Semantics

**Files:**
- Create: `meta_stackelberg/core/model_state.py`
- Create: `tests/meta_stackelberg/unit/test_model_state.py`

- [ ] **Step 1: Write failing tests for construction and update semantics**

Cover:

```python
old = ModelState.from_tensors([np.array([1.0, 2.0]), np.array([3.0])])
new = ModelState.from_tensors([np.array([2.0, 4.0]), np.array([1.0])])
delta = model_difference(new, old)
assert_arrays(delta, [[1.0, 2.0], [-2.0]])
assert_states_equal(apply_delta(old, delta), new)
assert_states_equal(apply_delta(old, delta, scale=0.5), [[1.5, 3.0], [2.0]])
```

Also test clone memory isolation, read-only tensors, flatten/unflatten, structure mismatch, non-floating tensor rejection, L2 norm, and cosine.

- [ ] **Step 2: Verify tests fail with import error**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_model_state.py -q`

Expected: FAIL because `model_state` is missing.

- [ ] **Step 3: Implement the minimal model-state API**

Required public API:

```python
class ModelState:
    @classmethod
    def from_tensors(cls, tensors: Iterable[np.ndarray]) -> 'ModelState': ...
    def clone(self) -> 'ModelState': ...
    def vector(self) -> np.ndarray: ...

def from_vector(vector: np.ndarray, template: ModelState) -> ModelState: ...
def model_difference(new: ModelState, old: ModelState) -> ModelState: ...
def apply_delta(old: ModelState, delta: ModelState, scale: float = 1.0) -> ModelState: ...
def state_l2_norm(state: ModelState) -> float: ...
def state_cosine(left: ModelState, right: ModelState) -> float: ...
```

Construction copies arrays and marks them read-only. All binary operations validate tensor count and shape.

- [ ] **Step 4: Run tests**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_model_state.py -q`

Expected: PASS.

## Task 3: Explicit RandomSource And Replay

**Files:**
- Create: `meta_stackelberg/core/random_state.py`
- Create: `tests/meta_stackelberg/unit/test_random_state.py`

- [ ] **Step 1: Write failing replay tests**

```python
source = RandomSource(seed=7)
snapshot = source.capture()
first = draw_all(source)
source.restore(snapshot)
second = draw_all(source)
assert first == second
```

`draw_all` draws from the owned Python `random.Random`, NumPy `Generator`, and Torch CPU `Generator`. Also test independent sources with the same seed and snapshot copy isolation.

- [ ] **Step 2: Verify missing-module failure**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_random_state.py -q`

Expected: FAIL because `random_state` is missing.

- [ ] **Step 3: Implement RandomSource**

Required API:

```python
@dataclass(frozen=True)
class RandomSnapshot:
    python_state: object
    numpy_state: dict
    torch_cpu_state: object | None

class RandomSource:
    def __init__(self, seed: int): ...
    @property
    def python(self) -> random.Random: ...
    @property
    def numpy(self) -> np.random.Generator: ...
    @property
    def torch(self): ...
    def capture(self) -> RandomSnapshot: ...
    def restore(self, snapshot: RandomSnapshot) -> None: ...
```

Torch is optional at import time; when installed, the source owns a CPU `torch.Generator`.

- [ ] **Step 4: Run replay tests**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_random_state.py -q`

Expected: PASS.

## Task 4: Federated Domain Records

**Files:**
- Create: `meta_stackelberg/federated/types.py`
- Create: `tests/meta_stackelberg/unit/test_federated_types.py`

- [ ] **Step 1: Write failing immutability and visibility tests**

Construct `ClientUpdate`, `RoundState`, `RoundRequest`, and `RoundTransition`. Verify metadata mappings are immutable, public signals and private diagnostics remain separate, and a transition preserves state-before/state-after without shared mutable arrays.

- [ ] **Step 2: Verify missing-module failure**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_federated_types.py -q`

Expected: FAIL because federated types are missing.

- [ ] **Step 3: Implement records**

```python
@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    delta: ModelState
    num_examples: int
    is_malicious: bool = False
    metadata: Mapping[str, Scalar] = field(default_factory=dict)

@dataclass(frozen=True)
class RoundState:
    round_index: int
    global_model: ModelState
    random_snapshot: RandomSnapshot
    component_states: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class RoundRequest:
    task_id: str
    state: RoundState
    sample_size: int
    server_lr: float = 1.0

@dataclass(frozen=True)
class RoundTransition:
    task_id: str
    state_before: RoundState
    sampled_clients: tuple[int, ...]
    benign_updates: tuple[ClientUpdate, ...]
    malicious_updates: tuple[ClientUpdate, ...]
    aggregate_delta: ModelState
    state_after: RoundState
    public_signals: Mapping[str, object]
    private_diagnostics: Mapping[str, object]
```

Reject non-positive `num_examples`, negative round indices, duplicate sampled clients, invalid sample size, and non-finite server learning rate.

- [ ] **Step 4: Run tests**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/unit/test_federated_types.py -q`

Expected: PASS.

## Task 5: Plug-In Protocols And Contract Tests

**Files:**
- Create: `meta_stackelberg/federated/protocols.py`
- Create: `tests/meta_stackelberg/contracts/test_federated_plugins.py`

- [ ] **Step 1: Write failing protocol tests**

Define scripted sampler, scripted trainer, mean aggregator, and server optimizer test doubles. Verify they satisfy runtime-checkable protocols and can be substituted independently.

- [ ] **Step 2: Verify missing-module failure**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/contracts/test_federated_plugins.py -q`

Expected: FAIL because protocols are missing.

- [ ] **Step 3: Implement runtime-checkable protocols**

```python
@runtime_checkable
class ClientSampler(Protocol):
    def sample(self, request: RoundRequest, rng: RandomSource) -> tuple[int, ...]: ...

@runtime_checkable
class LocalTrainer(Protocol):
    def train(self, client_id: int, state: RoundState, rng: RandomSource) -> ClientUpdate: ...

@runtime_checkable
class Aggregator(Protocol):
    def aggregate(self, updates: Sequence[ClientUpdate]) -> ModelState: ...

@runtime_checkable
class ServerOptimizer(Protocol):
    def step(self, model: ModelState, aggregate_delta: ModelState, learning_rate: float) -> ModelState: ...
```

- [ ] **Step 4: Run contract tests**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/contracts/test_federated_plugins.py -q`

Expected: PASS.

## Task 6: Sample-Weighted FedAvg And Server Update

**Files:**
- Create: `meta_stackelberg/federated/aggregation/fedavg.py`
- Create: `meta_stackelberg/federated/engine/server_optimizer.py`
- Create: `tests/meta_stackelberg/algorithms/test_fedavg.py`

- [ ] **Step 1: Write hand-computable failing tests**

Two updates `[1, 1]` with one example and `[3, 5]` with three examples must produce `[2.5, 4.0]`. Equal example counts must produce arithmetic mean. Empty updates, incompatible shapes, and non-positive total weight must fail.

Server SGD with model `[10, 20]`, aggregate delta `[2, -4]`, and learning rate `0.5` must produce `[11, 18]`.

- [ ] **Step 2: Verify missing implementation failure**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/algorithms/test_fedavg.py -q`

Expected: FAIL because implementations are missing.

- [ ] **Step 3: Implement FedAvg and ServerSGD**

FedAvg returns an aggregate delta and uses `num_examples` as weights. `ServerSGD.step` delegates to `apply_delta` and rejects non-finite or negative learning rates.

- [ ] **Step 4: Run algorithm tests**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/algorithms/test_fedavg.py -q`

Expected: PASS.

## Task 7: Deterministic Synthetic Clean Round

**Files:**
- Create: `meta_stackelberg/federated/engine/round_engine.py`
- Create: `tests/meta_stackelberg/integration/test_clean_round_engine.py`

- [ ] **Step 1: Write a failing end-to-end data-flow test**

Use injected scripted plugins:

```text
initial model [0, 0]
sample clients (0, 1)
client 0 delta [1, 1], examples 1
client 1 delta [3, 5], examples 3
FedAvg delta [2.5, 4.0]
server lr 0.5
next model [1.25, 2.0]
```

Assert every intermediate value in `RoundTransition`, not only the final model.

- [ ] **Step 2: Write failing replay and plugability tests**

Capture the initial state, run twice with restored RNG, and assert equal transitions. Replace FedAvg with a `FirstUpdateAggregator` and assert the engine uses the replacement without modification.

- [ ] **Step 3: Verify missing-engine failure**

Run: `/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/integration/test_clean_round_engine.py -q`

Expected: FAIL because `RoundEngine` is missing.

- [ ] **Step 4: Implement RoundEngine**

The engine performs only:

```text
sample -> local train -> aggregate -> server step -> transition
```

It records aggregate norm and sampled-client count as public signals. It does not evaluate a test set, compute reward, log artifacts, or mutate plugins outside their declared methods.

- [ ] **Step 5: Run integration and dependency tests**

Run:

```bash
/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/integration/test_clean_round_engine.py -q
/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg/regression/test_dependency_boundaries.py -q
```

Expected: PASS.

## Task 8: E0.1 Chinese Verification Report

**Files:**
- Create: `docs/milestones/E0.1-core-clean-round-report.zh-CN.md`

- [ ] **Step 1: Run the complete canonical suite**

Run:

```bash
/Users/antik/anaconda3/bin/python3 -m pytest tests/meta_stackelberg -q
git diff --check
```

Expected: all canonical tests pass and diff check is clean.

- [ ] **Step 2: Run the legacy regression suite**

Run:

```bash
/Users/antik/anaconda3/bin/python3 -m pytest meta_sg/tests tests -q
```

Expected: all legacy tests pass.

- [ ] **Step 3: Write the report**

The report records hypothesis, module contracts, exact data flow, hand-computable expected values, actual test results, plug-in replacement evidence, known limitations, and the E0.2 entry criteria.

- [ ] **Step 4: Self-review the report**

Verify the report contains no unsupported learning claims. This slice proves deterministic orchestration and math only; it does not prove neural-network FL convergence.

## Plan Self-Review

- The slice is independently runnable without datasets, downloads, checkpoints, or legacy imports.
- Every production behavior starts from a failing test.
- Update direction is fixed as `delta = local - global` and `global_next = global + lr * aggregate_delta`.
- Public signals and private diagnostics are separate records.
- Plugability is proven by replacing an aggregator without changing the engine.
- E0.1 does not claim attack validity, defense controllability, RL learning, or Meta-SG performance.
