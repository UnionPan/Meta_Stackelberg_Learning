# TD3 Attacker Replay Design Notes

These notes summarize the design discussion for training an adaptive attacker
with TD3 inside the Meta-Stackelberg FL setting.

## Core Transition

For an adaptive attacker, one FL round can be stored as:

```text
(obs_t, attack_action_t, attacker_reward_t, obs_{t+1}, done)
```

The attacker critic learns:

```text
Q_A(obs_t, attack_action_t)
```

More precisely, this is shorthand for:

```text
Q_A(obs_t, attack_action_t | defense policy, client sampling process,
     benign training dynamics, other attackers)
```

TD3 can still use `Q(obs, action)` if the other factors are either fixed as
part of the environment or represented inside `obs`.

## Is Replay Memory Meaningful?

Replay memory is meaningful because FL rollouts are expensive. A replay buffer
lets TD3 reuse collected FL transitions for many gradient updates.

However, replay quality depends on three things:

```text
1. obs contains the relevant defense/client context
2. the buffer does not mix too much stale non-stationary data
3. the reward matches the intended attack objective
```

If `obs` only contains compressed global weights, then replay can become noisy:
the same apparent state/action may lead to very different outcomes under
different sampled clients or defense behavior.

## Fixed Defense, Random Clients

When the defense policy is fixed, clients are usually treated as environment
randomness:

```text
Q_A(s, a_A | fixed defense, client sampling distribution)
```

This is valid if the attacker is meant to optimize expected attack performance
over client sampling.

Recommended setup:

```text
defense: fixed during attacker best-response update
client sampling: random each FL round
malicious set: fixed within an episode, resampled across episodes
replay: separate buffers per attack type, preferably recent or bucketed
```

The malicious set being fixed within an episode is useful because it models
compromised clients persisting for multiple FL rounds while still avoiding
overfitting across episodes.

## Why Random Client Sampling Makes Replay Hard

With random client sampling, non-IID data, and evolving global weights, exact
replay similarity is unlikely. Replay does not require identical samples, but
it relies on function approximation:

```text
Q(s, a) generalizes across nearby states/actions
```

If the state representation does not expose the relevant client/defense
context, there may be no meaningful notion of "nearby" for TD3 to exploit.

## Similarity Should Be Defense-Conditioned

A key point: two sampled client groups should not be considered similar merely
because their raw statistics are close. They are similar if they have similar
effects after the current defense/aggregator processes them.

Better definition:

```text
similarity = similarity of defense-conditioned effects
```

In other words, the attacker cares about whether its malicious update survives
the current defense and changes the aggregated model update.

## Unified Defense-Conditioned Embedding

Define a defense-conditioned context embedding:

```text
e_t = Phi_defense(client_updates, malicious_mask, defense_action)
```

Then replay similarity can be based on:

```text
distance(e_i, e_j)
```

instead of raw client IDs or raw client statistics.

A particularly useful feature is the attack residual after defense:

```text
attack_residual_after_defense =
    Aggr(benign_updates + malicious_updates)
  - Aggr(benign_updates_only)
```

This directly measures how much impact the malicious clients still have after
the defense. If two transitions have similar residuals, they are much more
similar from the attacker's point of view.

## Defense-Specific Similarity Features

Different defenses require different context features.

### FedAvg

FedAvg is mainly sensitive to the aggregate mean update.

Useful features:

```text
mean update direction
mean update norm
malicious client fraction
malicious update vs benign mean cosine
attack_residual_after_defense
```

### Norm Bounding / Clipping

For clipping defenses, raw norm is less important than clipped behavior.

Useful features:

```text
alpha
frac_benign_clipped
frac_malicious_clipped
mean_clipped_benign_norm
mean_clipped_malicious_norm
malicious_norm_over_alpha
clipped_attack_residual_norm
```

### Coordinate-Wise Trimmed Mean / Median

For trimmed mean and median, coordinate-level ranks matter.

Useful features:

```text
beta
malicious_coordinate_tail_rate
malicious_coordinate_survival_rate_after_trim
benign_coordinate_iqr_or_mad
malicious_coordinate_z_score
trimmed_mean_shift_norm
```

The most important question is whether malicious coordinates survive trimming.

### Krum / Multi-Krum

Krum is sensitive to pairwise update distances and cluster structure.

Useful features:

```text
malicious_to_benign_cluster_distance
benign_cluster_radius
nearest_neighbor_distance
krum_score
malicious_krum_rank
selected_by_krum
```

For Krum, looking only at update norm can be misleading. The attack succeeds
when malicious updates look close to the benign majority in distance space.

### FLTrust

FLTrust is sensitive to the root/server gradient direction.

Useful features:

```text
cosine_to_root_gradient
trust_score
malicious_trust_score_rank
normalized_update_direction
trust_weighted_attack_residual
```

## Current Meta-SG Defense Action

The current `meta_sg` defender action is:

```text
a_D = (alpha, beta, post_param)
```

where:

```text
alpha: norm bounding threshold
beta: coordinate-wise trimmed mean ratio
post_param: NeuroClip epsilon or Prun mask-rate parameter
```

Therefore, attacker observations and replay context should include at least:

```text
alpha
beta
post_param
frac_benign_clipped
frac_malicious_clipped
mean_clipped_benign_norm
mean_clipped_malicious_norm
malicious_coordinate_tail_rate
malicious_coordinate_survival_rate_after_trim
trimmed_mean_shift_norm
clean_acc_after_defense
backdoor_acc_after_defense
attack_residual_after_defense
```

## Replay Buffer Recommendations

For attacker TD3 in Meta-SG:

```text
1. Keep separate replay buffers per attack type.
2. Prefer recent replay over huge stale buffers.
3. Consider bucketed replay by defense-conditioned context.
4. Include defense action and defense diagnostics in attacker obs.
5. Avoid relying on raw client IDs; use permutation-invariant statistics.
6. For high-noise rewards, average over a few client-sampling seeds when affordable.
```

Possible buckets:

```text
round stage: early / mid / late
attacker fraction: low / medium / high
defense strength: weak / medium / strong
clip survival: low / high
trim survival: low / high
```

## Reward Notes

Untargeted attacker reward can be:

```text
r_A = -clean_acc
```

or smoother:

```text
r_A = clean_loss - stealth_penalty
```

Backdoor attacker reward should avoid rewarding trivial model destruction:

```text
r_A = backdoor_acc
      - lambda_clean * max(0, clean_acc_baseline - clean_acc)
      - lambda_stealth * stealth_penalty
```

Common stealth penalties:

```text
norm deviation from benign updates
cosine anomaly
low trust score
being trimmed/clipped/filtered
```

## Main Takeaway

The replay transition is useful, but the state must not be just `W_t`.

For TD3 attacker training in FL, the state should include:

```text
model features
defense action/features
client/update statistics
defense-conditioned attack residual
recent reward/action history
```

The important similarity notion is:

```text
not "same clients",
not "raw client stats are close",
but "the current defense turns these client updates into similar aggregate effects".
```
