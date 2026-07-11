# E0.2 Real Clean FL Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace E0.1 scripted components with deterministic data partitioning, client sampling, PyTorch model parameter conversion, real local SGD, sample-mean classification evaluation, and a no-download clean-FL convergence test.

**Architecture:** Each component implements a narrow contract and is tested without the round engine first. The integration test injects real implementations into the existing `RoundEngine`; it uses an in-memory linearly separable dataset so correctness does not depend on network access, MNIST caches, or historical artifacts.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.12, pytest.

---

## Task 1: Deterministic IID Partitioning

**Files:**
- Create: `meta_stackelberg/federated/data/__init__.py`
- Create: `meta_stackelberg/federated/data/partitioning.py`
- Create: `tests/meta_stackelberg/unit/test_partitioning.py`

- [ ] Write tests that partition 20 indices across four clients, asserting complete coverage, no overlap, near-equal sizes, same-seed equality, different-seed difference, and rejection when samples are fewer than clients.
- [ ] Run the test and verify missing-module failure.
- [ ] Implement `iid_partition(num_samples, num_clients, rng) -> tuple[tuple[int, ...], ...]` using the injected NumPy generator and `np.array_split`.
- [ ] Run the focused tests and expect PASS.

## Task 2: Uniform Client Sampler

**Files:**
- Create: `meta_stackelberg/federated/clients/__init__.py`
- Create: `meta_stackelberg/federated/clients/sampling.py`
- Create: `tests/meta_stackelberg/unit/test_client_sampling.py`

- [ ] Write tests for deterministic without-replacement sampling, sample-size validation, and runtime satisfaction of `ClientSampler`.
- [ ] Verify missing-module failure.
- [ ] Implement `UniformClientSampler(num_clients)` using only `request.sample_size` and `rng.numpy.choice`.
- [ ] Run focused tests and expect PASS.

## Task 3: PyTorch Parameter Codec

**Files:**
- Create: `meta_stackelberg/federated/models/__init__.py`
- Create: `meta_stackelberg/federated/models/parameters.py`
- Create: `tests/meta_stackelberg/unit/test_torch_parameters.py`

- [ ] Write tests using `torch.nn.Linear(2, 2)` for capture/load round trip, input-copy isolation, parameter count mismatch, shape mismatch, and dtype preservation.
- [ ] Verify missing-module failure.
- [ ] Implement `TorchParameterCodec.capture(model)` and `load(model, state)`. The codec handles trainable parameters only; model buffers are explicitly outside E0.2 scope.
- [ ] Run focused tests and expect PASS.

## Task 4: Real Local SGD Trainer

**Files:**
- Create: `meta_stackelberg/federated/clients/trainer.py`
- Create: `tests/meta_stackelberg/unit/test_local_trainer.py`

- [ ] Write tests using a four-sample binary dataset and zero-initialized linear model. Assert returned update uses `local - global`, `num_examples` is correct, training metrics are finite, global state is unchanged, replay from the same RNG snapshot is exact, and unknown client ids fail.
- [ ] Verify missing-module failure.
- [ ] Implement `TorchLocalTrainer(model_factory, client_datasets, codec, learning_rate, local_epochs, batch_size)`. Every call constructs a fresh model and optimizer, loads the global state, and uses the injected Torch generator in `DataLoader`.
- [ ] Run focused tests and expect PASS.

## Task 5: Canonical Classification Evaluator

**Files:**
- Create: `meta_stackelberg/federated/evaluation/__init__.py`
- Create: `meta_stackelberg/federated/evaluation/classification.py`
- Create: `tests/meta_stackelberg/unit/test_classification_evaluator.py`

- [ ] Write tests proving loss/accuracy are invariant across batch sizes 1, 2, 4, and 8, empty datasets return count zero with finite zero metrics, and evaluation does not mutate the supplied state.
- [ ] Verify missing-module failure.
- [ ] Implement `ClassificationMetrics(loss, accuracy, num_examples)` and `ClassificationEvaluator.evaluate(state)` with sample-weighted cross entropy.
- [ ] Run focused tests and expect PASS.

## Task 6: Real Clean-FL Convergence Gate

**Files:**
- Create: `tests/meta_stackelberg/integration/test_real_clean_fl.py`
- Create: `docs/milestones/E0.2-real-clean-fl-report.zh-CN.md`

- [ ] Build an in-memory, linearly separable two-class dataset; partition it across four clients with the canonical partitioner.
- [ ] Create a zero-initialized `Linear(2, 2)` model, canonical `RoundState`, `UniformClientSampler`, `TorchLocalTrainer`, `FedAvg`, `ServerSGD`, and `ClassificationEvaluator`.
- [ ] Run ten rounds selecting all clients. Assert final loss is below initial loss, final accuracy is at least 95%, every transition contains four client updates, and rerunning from the same initial state produces identical final parameters and metric history.
- [ ] Run all canonical tests, legacy tests, dependency guard, and `git diff --check`.
- [ ] Write a Chinese report with module contracts, full data flow, expected/actual convergence, replay evidence, limitations, and E0.3 entry criteria.

## Expected Data Flow

```text
TensorDataset
 -> iid_partition
 -> client Dataset subsets
 -> UniformClientSampler
 -> TorchLocalTrainer per selected client
 -> ClientUpdate(delta=local-global, num_examples)
 -> FedAvg
 -> ServerSGD
 -> RoundState_next
 -> ClassificationEvaluator
 -> loss/accuracy history
```

## Expected Result

The zero-initialized classifier begins near cross entropy `log(2)` and accuracy `0.5`. On the linearly separable dataset, ten canonical FL rounds should reduce loss and reach at least `0.95` accuracy. Exact numerical loss is not hard-coded; direction, threshold, transition contents, and replay equality are the gate.

## Plan Self-Review

- No test downloads data or reads legacy outputs.
- Every component is tested before integration.
- `RoundEngine` is reused unchanged, proving the scripted trainer can be replaced by a real trainer.
- Evaluation remains outside the round state transition.
- Model buffers, CUDA determinism, non-IID partitioning, and MNIST convergence are explicitly deferred to later E0 slices and are not claimed here.
