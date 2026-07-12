# MNIST White-Box Backdoor Meta-SG Design

## 1. Objective and authority

This design adds a paper-aligned MNIST white-box backdoor experiment without changing the existing
untargeted Meta-SG protocol. Its authority is the user-supplied *Meta Stackelberg Game* passages on
space compression, synthesized defenses, self-generated data, white-box evaluation, and online adaptation,
plus the cited RL-backdoor and federated-pruning methods.

Executable RL-backdoor provenance is pinned to `HengerLi/RLBackdoorFL` commit
`aebb9e96f13f9a848932fed214be8058c5d19854`; in particular,
`train_cifar_post_DDPG_policy.py`, `mnist_test.py`, and `DataProcess.py`. Paper text takes precedence over
repository smoke-script defaults when they differ, and each such difference is listed below.

The implementation has two ordered milestones:

1. **White-box vertical slice:** the server knows the real trigger and target label and uses real MNIST
   client data. This is an upper-bound experiment and proves the BRL/Defender/Algorithm 1 path.
2. **Generated-data experiment:** start from the original 200 seed samples with `q=0.1`, train the referenced
   MNIST cGAN directly for 100 epochs without augmentation, generate 60,000 samples, and repeat the same
   protocol. The paper's 5,000-sample augmentation path is a later matched reproduction, not a dependency of
   the first generated-data run.

Generated-data results must not be labeled white-box upper-bound results. CIFAR-10 and unknown-trigger
GAN experiments are outside this first implementation cycle.

## 2. Fixed MNIST white-box task

The first vertical slice uses one immutable task definition:

```text
dataset: MNIST
knowledge: true client data + true trigger + true target label
source label: digit 1
target label: digit 7
trigger: the cited MNIST global-trigger fixture
poison ratio: 0.5
backdoor attackers: 5 of 100 clients
client subsampling rate: 10%
```

The source/target labels, poison ratio, and global-trigger role follow *Learning to Backdoor Federated Learning*.
The public `RLBackdoorFL` MNIST smoke script uses `0.4`; that repository difference is recorded but does not
override the paper-aligned `0.5` primary configuration. A `0.4` run may appear only as a named reproduction
ablation.
The exact Figure-3 global-trigger pixels must be captured as a checked-in immutable fixture and recorded by pixel
coordinates, values, shape, and hash in every manifest. The held-out clean and triggered query sets never enter
replay, reward fitting, checkpoint selection, or cGAN training. In white-box Phase A, the server reward set is
an authorized read-only view of true client training examples; those examples remain assigned to clients and are
not removed from the 60,000-example FL training set.

The existing source-class poisoning/evaluator is the correct task family and is extended only where required to
enforce the exact global-trigger fixture and isolated held-out query role. DBA sub-trigger behavior remains a
separate baseline and is not used by the primary BRL task.

## 3. BSMG information and state compression

One Markov step remains one FL round. Client sampling occurs before the Attacker action, and all malicious
clients controlled by the same Attacker execute the same three-dimensional action.

The physical state is the full global model. Policy inputs use the stable flattened representation of the final
two learnable tensors in the model weight list, normalized by the existing finite normalizer. This is the current
executable interpretation of the paper's “last two hidden layers” wording and yields 1,290 MNIST model-tail
features (`fc1.weight`, `fc1.bias`). The manifest records tensor names and shapes so this interpretation is not
silently presented as an undisputed paper constant.

```text
Defender observation = model_tail + round_progress
Attacker observation = model_tail + round_progress
                       + sampled_malicious_count + current_Defender_raw_action
```

The Defender is not given the attack-type label. Trigger and target knowledge enter the white-box reward,
poisoned training data, and post-defense evaluator; they are fixed for the experiment and are not extra policy
actions.

## 4. Defender action and execution

The TD3 Actor output is always exactly three-dimensional:

```text
raw action in [-1,1]^3
a_D^t = (alpha_t, beta_t, epsilon_t / sigma_t)
```

The configured post-defense branch determines the third coordinate's meaning; method selection is not an
additional learned action.

### 4.1 Shared training-stage defenses

1. Normalize each sampled client update to the L2 threshold
   `alpha in (0, max_i ||g_i^t||]`.
2. Apply coordinate-wise symmetric trimmed mean with `beta`. The paper states `beta in [0,1)`; the executable
   codec must enforce the per-round feasibility constraint `2 * floor(beta*n) < n`. A global `0.45` cap may be
   retained only as an implementation declaration.

### 4.2 NeuroClip branch

```text
(alpha, beta, epsilon)
```

`epsilon` is decoded to a declared positive clip range and applied to a post-defense model copy. The intermediate
global FL state remains the aggregation output; the defended copy supplies reward/evaluation and final delivery.

### 4.3 Prun branch

```text
(alpha, beta, sigma)
```

Prun follows Wu et al., *Mitigating Backdoor Attacks in Federated Learning* rather than magnitude pruning.
Using the server's clean white-box data, it ranks redundant/dormant neurons or channels according to the cited
activation-based procedure, masks the fraction prescribed by `sigma`, and applies the reference extreme-weight
adjustment where applicable. The pruned object is a copy; masks never mutate the aggregated training state.
The first implementation must expose the scored layer set, ordering, tie-break, realized mask count, and whether
fine-tuning/weight adjustment is enabled. A simplified mask is allowed only as an explicitly named ablation.

NeuroClip and Prun runs are separate primary evidence records. No four-dimensional
`(alpha,beta,epsilon,sigma)` action and no hidden discrete method selector are permitted.

## 5. White-box backdoor Attacker

The backdoor follower is a separate type-specific TD3 policy, not the existing untargeted local-search policy.
It implements the Meta-SG paper's compressed three-dimensional real action and applies the same action to every
sampled malicious device. The authoritative executable contract is the three-dimensional post-defense DDPG
environment in the public `RLBackdoorFL` repository:

```text
a_A^t = (rho_t, eta_t, E_t)
rho: poisoned source-class fraction, decoded on {0.0, 0.1, ..., 1.0}
eta: malicious local learning rate, decoded to [0.0, 0.1]
E: malicious local epochs, decoded to {1, ..., 10}
```

The repository's raw-action formulas contain endpoint-sensitive integer conversions, so the new codec defines
and tests explicit clipping at both endpoints rather than inheriting Python truncation accidentally. Every
sampled malicious client trains from the same pre-round global state with the same decoded controls but its own
deterministically seeded local data order.

This compressed variant disables the fourth model-scaling coordinate used by the repository's separate Krum/TD3
experiment and does not expose the full Double Whammy selective-crafting controls. Those richer controls are a
separate attack ablation, not hidden fixed coordinates in the primary Meta-SG action. The codec may not reuse the
untargeted `(gamma,E,lambda)` semantics.

The production acceptance test compares each decoded coordinate against the checked-in adapter contract and runs
the same raw action twice under fixed seeds to obtain identical poisoned indices, local steps, local batches, and
malicious updates. Any bounds not printed by Meta-SG are versioned implementation declarations.

The initial backdoor attack domain contains separately pretrained BRL policies targeted against the paper's
specialized defenses:

```text
BRL pretrained against fixed Norm Bounding
BRL pretrained against fixed NeuroClip
BRL pretrained against fixed Prun (when the Prun branch is enabled)
```

Each policy receives its own 300-FL-round sequential TD3 pretraining run, replay, optimizer/RNG state, origin
label, and round-boundary checkpoint. `K` is the number sampled from `Q(Xi)` per Algorithm 1 iteration, not the
number of artifacts.

## 6. Reward and metrics

White-box reward uses a server-owned training/evaluation split derived only from authorized real client data.
For each post-defense copy, compute:

```text
L_clean  = cross entropy on clean white-box reward data
L_safe   = cross entropy on triggered digit-1 examples with original label 1
L_target = cross entropy on the same triggered examples relabeled to target 7
```

The Defender maximizes a declared clean/backdoor trade-off:

```text
r_D = -[(1-lambda_D) * L_clean + lambda_D * L_safe]
```

The BRL Attacker maximizes target success while retaining main-task stealth:

```text
r_A = -(1-lambda_A) * L_target - lambda_A * clean_damage_penalty
```

For the BRL objective, the cited source defines
`F_attack = lambda * F_clean + (1-lambda) * F_poisoned` with `lambda=0.5`; the Attacker maximizes its negative.
Defender and Attacker coefficient meanings are stored separately to prevent an accidental sign/role alias. The
Defender's `L_safe` term uses original label 1 and is therefore not the Attacker's poisoned-label loss.

Every transition records scalar reward and its clean/target/safe components. Held-out evidence separately reports:

- clean loss and clean accuracy;
- attack success rate over every held-out non-target example;
- source-example count and successes;
- post-defense realized action and pruning/clipping summary.

ASR alone cannot pass a Gate: a model that destroys clean accuracy is not a successful defense.

## 7. Algorithm 1 Reptile Meta-SG

The primary training output is Algorithm 1 with its practical Reptile branch; Algorithm 2 is an optional meta-RL
baseline and is not chained after Algorithm 1.

```text
for n_D in 0 .. N_D-1:
    sample K BRL tasks uniformly from Q(Xi)
    for each sampled task xi:
        adapt Defender once: theta -> theta_xi
        freeze theta_xi
        update the same BRL policy N_A times -> phi_xi(N_A)
        freeze phi_xi(N_A)
        update task Defender at theta_xi -> theta_bar_xi
    theta <- theta + (1/K) * sum(theta_bar_xi - theta)
```

The Attacker BR is conditioned on the adapted Defender `theta_xi`. Full TD3 policy/replay fingerprints enforce
both freeze directions. Paper parameters remain `K=10`, `N_A=10`, `N_D=10`, `eta=0.01`, and
`kappa_A=kappa_D=0.001`, with `H=200` for MNIST trajectories. Scaled runs may reduce counts but not nesting,
roles, action semantics, reward permissions, or task identity.

## 8. Data paths

### 8.1 Phase A: real-data white-box upper bound

Assign all 60,000 real MNIST training examples exactly once across 100 clients; under the default IID setup this
is approximately 600 examples per client. Construct the white-box reward dataset as a deterministic, stratified,
read-only server view into those same true training examples. This overlap is intentional and permitted only by
the white-box threat model: reward access does not remove, duplicate, or reassign any client example. The 10,000
official test examples form the held-out query dataset and never enter reward, replay, policy updates, checkpoint
selection, or client training. The manifest records all client partitions and reward-view indices, and the run
protocol is named `mnist-whitebox-real-data-v1`.

### 8.2 Phase B0: original-200 generated-data experiment

Use the original 200 initial samples with `q=0.1` exactly once as the cGAN training dataset. Normalization that
belongs to the model input pipeline remains enabled, but random rotation, color jitter, geometric augmentation,
synthetic duplication to 5,000, and augmentation-based resampling are disabled. Train the referenced conditional
MNIST GAN for 100 epochs with its default network parameters, versioning the reference commit/config. Generate
exactly 60,000 labeled images—6,000 requested samples per class—with a declared generator seed. The real held-out
MNIST query data remains isolated.

Because 200 examples may be insufficient for a stable cGAN, Phase B0 is an explicit small-data experiment rather
than an assumed replacement for the paper result. Evidence must report seed-set class counts, per-class generated
counts, discriminator/generator losses, deterministic sample grids, diversity statistics, and downstream clean
accuracy/ASR. Mode collapse or missing class support is a scientific result and must not be repaired by silently
adding augmented examples.

Artifacts include generator/discriminator checkpoints, the original-seed index manifest, loss history,
sample-grid QA, class histogram, image range/shape validation, and dataset hash. A typed generated-data bundle is
accepted by the same environment factory used in Phase A.

### 8.3 Phase B1: paper augmentation reproduction

After Phase B0, a separate matched run may augment the same 200 seeds to 5,000 examples using normalization,
random rotation, and color jitter before the same 100-epoch cGAN training and 60,000-sample generation. Phase B1
uses separate artifact identities and is compared against B0 under the same GAN seed, architecture, optimizer,
generation class counts, FL budget, and query seeds. B1 is not required to begin or complete B0.

## 9. Online adaptation

Online execution starts from the Algorithm 1 meta-policy and never reinitializes it. Each real FL round executes
one Defender action and stores `(s,a,r_tilde,s')`. For MNIST, collect 50 FL rounds before a policy-update block,
then update the same TD3 policy/replay while preserving continuity. This window replaces the older 100-round
MNIST setting in the current ledger for this paper version.

Phase A estimates online reward using authorized white-box data. Phase B uses self-generated data plus any
separately authorized inferred/augmented data; held-out query data is forbidden. `N_A` best-response training is
an offline Algorithm 1 operation and is not rerun every online FL round.

## 10. Evidence and Gates

All comparisons use matched FL-round, trajectory, TD3-update, and query-seed budgets. Primary white-box evidence
must establish:

1. BRL changes under a frozen adapted Defender or reaches a preregistered oracle plateau.
2. Different Defender commitments induce different BRL responses.
3. Few-shot Defender adaptation improves the joint clean/backdoor objective.
4. Algorithm 1 initialization beats random initialization and no-adaptation baselines under equal budgets.
5. Specialized-oracle regret is below a preregistered threshold.
6. Clean accuracy remains above its floor while ASR improves by its required margin.
7. NeuroClip and Prun action dimensions produce measurable, correctly attributed operator effects.

Smoke runs prove execution only. A performance claim requires real MNIST white-box data, frozen artifacts,
independent held-out query seeds, preregistered thresholds, and no query-driven tuning.

## 11. Architecture and isolation

Add separate backdoor components rather than branching the untargeted environment internally at every step:

```text
shared: TD3, replay/checkpoint, FL model/state, Algorithm 1, task sampler
backdoor-specific: source-1-to-target-7 poisoning, BRL action/operator, white-box reward,
                   NeuroClip/Prun post-defense adapter, targeted evidence/Gates
untargeted-specific: current local-search Attacker and untargeted reward
```

The environment consumes protocols for attack generation, reward evaluation, and post-defense transformation.
This prevents white-box permissions or triggered query data from leaking into untargeted experiments.

## 12. Failure handling and provenance

- Reject any query overlap with client, reward, cGAN, replay, or checkpoint-selection data. In white-box Phase A,
  require reward indices to be a declared subset of client training indices; reject this overlap in non-white-box
  protocols.
- Reject trigger/target mismatch between poisoned clients, evaluator, reward, and checkpoint.
- Reject post-defense kind/action-decoder mismatch.
- Reject infeasible per-round trimming and invalid realized pruning counts.
- Reject restoration under different model schema, task identity, reward protocol, or defense branch.
- Persist every paper value, implementation-declared bound, seed, dataset/trigger hash, action trace, update count,
  and fingerprint in atomic artifacts.

## 13. Delivery order

1. Typed digit-1-to-digit-7 white-box task, global-trigger poisoning/evaluation, and joint reward.
2. Paper action refactor with NeuroClip/Prun branch semantics.
3. Reference-conformant Prun operator.
4. BRL action/operator and specialized attack-domain pretraining.
5. MNIST white-box BSMG environment and Algorithm 1 runner.
6. White-box evidence/Gates and real-data scaled run.
7. Online 50-round collection/update blocks.
8. Original-200 MNIST cGAN training, 60,000-sample bundle, and generated-data comparison.
9. Optional matched 5,000-example augmentation reproduction.

The first implementation cycle is complete only after Phase A runs end-to-end with frozen white-box artifacts and
scientific evidence. Phase B0 is a required subsequent milestone, not a substitute for Phase A verification;
Phase B1 is the later paper-augmentation reproduction.
