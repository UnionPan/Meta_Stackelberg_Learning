# Training-frozen Histogram40

[distributions.py](distributions.py) provides Histogram40 fitting/encoding and the original schema-version-1 JSON format.
Fitting consumes explicitly supplied training C/N/B/Z matrices. Policy training, frozen evaluation, and resume only encode with the fitted values.

| Observation component | Width | Summary |
| --- | --- | --- |
| `histogram40` | 40 | Ten probability bins per feature; nine training-fitted boundaries |

Histogram boundaries are per-feature P10…P90 quantiles pooled over training clients. Outer bins are unbounded, values on a boundary enter its right bin, and repeated boundaries leave empty bins without shifting them.
The 40-dimensional output is flattened in bin-major order: each row contains C, N, B, Z. This is different from the older simplified `histogram` component's feature-major, fixed-range layout.

The per-round raw summaries are computed as float32, then training mean/std are fitted in float64. Standard deviations have a `1e-4` floor. Deployment values use those frozen parameters and are returned as float32.
`--no-standardize` stores zero means and unit scales while retaining the fitted histogram boundaries.

## Fit from training contexts

RL round metrics now retain `aggregation.client_features` (float32 raw C/N/B/Z) and `client_norms`. A calibration run can use `rl.mode="fixed"` and the ordinary `compact8` observation on the selected training task. Save it in its own directory, then fit from those training-run metrics:

```bash
meta-sg --fit-observation histogram40 \
  --calibration outputs/calibration_run/metrics.jsonl \
  --output outputs/transforms/histogram40.json
```

The fitter ignores round zero and reads only raw client features; test ACC/ASR fields do not participate. To use historical signed log norms, add `--signed-log-norm`, which computes N from the logged update norms. Configure the runtime observation with the same `signed_log_norm` value.

Alternatively, prepare an NPZ with `training_features` of shape `[training_rounds, sampled_clients, 4]`. Its columns must be the intended raw C/N/B/Z features, not an eight-dimensional summary or evaluation observations:

```bash
meta-sg --fit-observation histogram40 \
  --calibration calibration/training_contexts.npz \
  --output outputs/transforms/histogram40.json
```

`--calibration-key raw_client_features` reads the raw feature array name used by the old training archives.
This command only fits statistics; it does not load image datasets or run FL/RL training. The file records the source hash, selected feature field, training round/client counts, frozen statistics, and the norm convention when converting metrics. It requires a new output file.
Choose training-only calibration data explicitly. The command does not infer whether an externally supplied array came from training or held-out evaluation.

For a Python list of training matrices, including different client counts per round:

```python
from meta_sg.rl.distributions import DistributionTransform

transform = DistributionTransform.fit(training_clients, 'histogram40')
transform.save('histogram40.json')
observation = transform.encode(current_clients)
```

Existing schema-version-1 Histogram40 transform files load directly through `DistributionTransform.load()` or the JSON configuration below.

## Select in JSON

Set `rl.observation` to:

```json
{
  "name": "histogram40",
  "parameters": {
    "transform_path": "outputs/transforms/histogram40.json",
    "signed_log_norm": false
  }
}
```

[The focused MNIST example](../../../configs/mnist_rl_histogram40.json) supplies the experiment settings; omitted fields use the shared defaults. Fit or supply the referenced transform before starting training.
The component variant must match the saved transform. `--dry-run` reports the width without opening the transform file or starting training.

`signed_log_norm=false` (default) preserves the shared extractor's absolute client/root log norm ratio.
Set it to true when the calibration archive used the historical signed log-update-norm channel; the observation then replaces N with `log(max(update_norm, float64_tiny))` and casts raw features to float32, matching the historical feature precision.
The example uses the shared absolute-ratio channel; enable the signed channel when matching the corresponding old experiment. The calibration archive and runtime must use the same feature definition. Metrics-fitted transforms check this choice; old NPZ/JSON artifacts without that metadata rely on the caller's explicit setting.
This setting changes the observation only; the existing risk/action calculations are unchanged. The new feature extractor and risk scorer are not a claim of whole-experiment historical reproduction.

## Persistence

The run's `observation.json` stores a copy of the frozen transform. `latest.pt` embeds the same state alongside policy/replay/RNG state, and `policy.json` embeds it with the frozen policy export.
Resume and frozen inference use the embedded transform, so the original `transform_path` file can be unavailable after the initial snapshot. No deployment observations are used to refit edges or normalization.
Custom stateless observation functions continue to use the existing registry contract. The built-in frozen distribution wrapper caches its loaded transform and serializes that state explicitly.
