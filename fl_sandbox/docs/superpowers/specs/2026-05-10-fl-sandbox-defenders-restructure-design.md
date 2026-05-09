# fl_sandbox Defenders Restructure Design

## Goal

Make `fl_sandbox.defenders` the public defender adapter API, mirroring the
new `fl_sandbox.attacks` boundary and removing defender code from
`fl_sandbox.core`.

## Current State

`fl_sandbox.aggregators` already owns the aggregation algorithms:
`AggregationDefender`, `PaperActionDefender`, and the concrete aggregation
functions. `fl_sandbox.core.defender` owns the runtime adapter layer:
`SandboxDefender`, typed defender wrappers, `create_defender`,
`supported_defense_types`, and `build_defender_config_kwargs`.

That adapter layer is not core runtime infrastructure, so keeping it under
`core` blurs the package boundary in the same way old attacker paths did.

## Target Layout

```text
fl_sandbox/
  attacks/       # attacker public API and implementations
  defenders/     # defender public API, wrappers, config factory
  aggregators/   # pure aggregation algorithms
  core/          # runtime, experiment builders, metrics, postprocess
```

`fl_sandbox.defenders` exposes:

- `SandboxDefender`
- `FedAvgDefender`, `KrumDefender`, `MultiKrumDefender`
- `MedianDefender`, `ClippedMedianDefender`, `GeometricMedianDefender`
- `TrimmedMeanDefender`, `PaperNormTrimmedMeanDefender`, `FLTrustDefender`
- `DEFENSE_CHOICES`
- `create_defender`
- `supported_defense_types`
- `build_defender_config_kwargs`
- aggregation runtime re-exports currently used by callers, including
  `AggregationDefender`, aggregate functions, `weights_to_vector`, and
  `vector_to_weights`

## Import Rules

External and in-repo runtime code imports defender adapters from
`fl_sandbox.defenders`.

Runtime aggregation algorithms continue to be available from
`fl_sandbox.aggregators`. Existing direct imports from `fl_sandbox.aggregators`
should stay as-is.

No compatibility shim remains at `fl_sandbox.core.defender`. All in-repo imports
from `fl_sandbox.core.defender` are migrated or removed, and `fl_sandbox.core`
stops lazily re-exporting defender symbols.

## Scope

In scope:

- Move `core/defender` adapter modules to `fl_sandbox/defenders`.
- Update imports in `fl_sandbox`, `meta_sg`, and tests.
- Update tests to assert the new public API and reject old defender imports.
- Delete `fl_sandbox/core/defender`.
- Update README folder layout.

Out of scope:

- Rewriting aggregation algorithms.
- Moving `fl_sandbox.aggregators`.
- Changing defender behavior or default parameters.
- Adding Tianshou or changing RL attacker internals.

## Testing

Tests must prove:

- `from fl_sandbox.defenders import create_defender, DEFENSE_CHOICES` works.
- `build_defender_config_kwargs` still builds `SandboxConfig` kwargs,
  including `paper_norm_trimmed_mean`.
- Old imports from `fl_sandbox.core.defender` are absent from Python source.
- Full test suite passes.

