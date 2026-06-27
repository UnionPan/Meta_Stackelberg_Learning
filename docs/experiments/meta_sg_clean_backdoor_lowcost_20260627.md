# Meta-SG Clean + Backdoor Low-Cost Experiment

Date: 2026-06-27

Run directory:

`runs/meta_sg_goal/clean_backdoor_lowcost/20260627-110402`

## Goal

Run a low-cost Meta-SG validation focused on clean and targeted backdoor tasks before expanding to the full attack domain. The experiment covers:

- clean
- BFL
- DBA
- RL-backdoor

The purpose was to check whether clean/backdoor task-aware meta-training plus few-shot adaptation gives a useful H50 defense signal.

## Training Setup

Command family: `meta_sg/scripts/run_meta_sg_pretraining.py`

Key settings:

- backend: `fl_sandbox`
- dataset: `mnist`
- attack domain: `clean_backdoor`
- task sampler: `stratified`
- meta objective: `query_targeted_reptile`
- T/K/H/l: `5 / 4 / 10 / 10`
- query horizon: `10`
- defender action: `server_lr`
- server lr range: `0.4..1.0`
- lambda_bd: `1.0`
- clients/attackers/subsample: `20 / 4 / 0.2`
- client/eval samples: `64 / 500`
- device: `cuda:0`
- seed: `2026`

Training summary:

- mean defender reward last 10: `0.3554116757084064`
- final checkpoint: `runs/meta_sg_goal/clean_backdoor_lowcost/20260627-110402/final`
- query accept rate during training was usually low, mostly `0.25`, with one `0.00` iteration.

## H50 Base Evaluation

Artifact:

`runs/meta_sg_goal/clean_backdoor_lowcost/20260627-110402/direct_h50_backdoor_base.json`

| Scenario | Clean Acc | ASR | Defense Score |
| --- | ---: | ---: | ---: |
| clean | 0.892 | 0.000 | 0.892 |
| BFL | 0.876 | 0.062 | 0.814 |
| DBA | 0.884 | 0.002 | 0.882 |
| RL-backdoor | 0.898 | 0.090 | 0.808 |

Targeted aggregate:

- mean clean: `0.886`
- min clean: `0.876`
- mean ASR: `0.051333333333333335`
- max ASR: `0.09`
- mean defense score: `0.8346666666666668`

## Few-Shot Physical-Target Adaptation

### Selection Horizon 10 + Deployment Clean Recovery

Artifact:

`runs/meta_sg_goal/clean_backdoor_lowcost/20260627-110402/direct_h50_backdoor_physical_target.json`

Result: no candidate was accepted for BFL, DBA, or RL-backdoor.

Reason: the short support/query horizon under-detected the backdoor. Query ASR was only:

- BFL: `0.004`
- DBA: `0.002`
- RL-backdoor: `0.006`

Final H50 metrics therefore stayed identical to base.

### Selection Horizon 50 + Score Selection

Artifact:

`runs/meta_sg_goal/clean_backdoor_lowcost/20260627-110402/direct_h50_backdoor_physical_target_sel50.json`

| Scenario | Base Clean / ASR | Adapted Clean / ASR | Selected Target |
| --- | ---: | ---: | --- |
| BFL | 0.876 / 0.062 | 0.812 / 0.002 | alpha=0.10, beta=0.38 |
| DBA | 0.884 / 0.002 | 0.834 / 0.000 | alpha=0.12, beta=0.38 |
| RL-backdoor | 0.898 / 0.090 | 0.844 / 0.000 | alpha=0.10, beta=0.38 |

Targeted aggregate after adaptation:

- mean clean: `0.83`
- min clean: `0.812`
- mean ASR: `0.0006666666666666666`
- max ASR: `0.002`
- mean defense score: `0.8293333333333334`

## Interpretation

This run shows two distinct facts:

1. The base low-cost clean/backdoor Meta-SG checkpoint already keeps H50 targeted ASR relatively low in this small setting, with max ASR `0.09`.
2. Full-horizon physical-target adaptation can almost eliminate ASR, but it hurts clean accuracy too much.

The main failure is not simply "no defense against backdoor"; it is that the current adaptation is too much like a strong backdoor suppressor and not enough like a support-conditioned Meta-SG adapter. Short support horizons miss delayed backdoor signals, while long-horizon physical targets over-defend and reduce clean accuracy.

## Next Experimental Gap

The next proof target should be:

`support trajectory -> infer attack family -> choose clean-constrained adaptation -> improve H50 score`

Minimum next comparisons:

- base meta-policy
- pretrain-only / no-adaptation
- random-init + adaptation
- oracle attack-label adapter
- support-only inferred adapter
- physical-target rule

Metrics to report:

- H50 clean mean/min
- H50 ASR mean/max
- attack-family inference accuracy
- multi-seed mean +/- std
- adaptation wall time
