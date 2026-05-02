# Evaluating Meta-SG Policies

There are two separate questions:

1. Did training converge?
2. Is the learned defender policy effective?

Do not answer either question with training reward alone.

## Convergence

Use the training curve only as a stability diagnostic. The current helper
`assess_convergence()` checks the recent reward window for:

- low rolling standard deviation
- near-zero linear slope
- enough iterations

Example:

```python
from meta_sg.learning.evaluation import assess_convergence

report = assess_convergence(result.defender_rewards, window=10)
```

This says whether optimization has flattened under the current environment. It
does not prove the policy is robust.

## Effectiveness

Evaluate the frozen policy on held-out rollouts and compare it against
baselines. At minimum compare:

- learned meta defender
- neutral constant action
- strong clipping / strong defense action
- fixed hand-designed defenses once the `fl_sandbox` adapter is connected

Report:

- mean defender reward
- worst-case defender reward across attack types/seeds
- final clean accuracy
- final backdoor accuracy
- variance across seeds

The learned policy is useful only if it improves robust utility without hiding a
bad clean/backdoor tradeoff.

## Small Stub Evaluation

Run:

```bash
conda run -n gym_env python meta_sg/scripts/evaluate_small.py
```

This writes TensorBoard logs under:

```text
runs/meta_sg_small_eval/<timestamp>
```

The stub environment is only a workflow test. It now contains an
action-dependent proxy tradeoff so baselines differ, but it is not evidence of
real FL robustness.

## Real FL Evaluation With `fl_sandbox`

`meta_sg.simulation.fl_sandbox_adapter.FLSandboxCoordinatorAdapter` connects the
paper-shaped game interface to `fl_sandbox.federation.runner.MinimalFLRunner`.

The adapter:

- exposes static `spec` without resetting
- maps paper defense action `(alpha, beta, epsilon/sigma)` to per-round
  norm-bound + trimmed-mean aggregation
- keeps post-training defense outside the FL transition
- translates sandbox `RoundSummary` into the `meta_sg` summary type
- caches the last full evaluation when `eval_every` skips a round

Use this adapter for real FL validation after the stub workflow is healthy.
Start with very small settings, because every game step is a real FL round.
