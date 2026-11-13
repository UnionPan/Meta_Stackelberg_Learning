"""Design: share weighted-delta application across configurable action mechanisms.
Data flow: risks and client/root norms + normalized action -> soft weights ->
blend median clipping and trusted-root normalization -> weighted parameter delta.
"""

import numpy as np


def weighted_update(updates, context, weights, scales):
    total = weights.sum()
    update = tuple(sum(float(weight * scale / total) * part
                       for weight, scale, part in zip(weights, scales, parts, strict=True))
                   if total > 0 else parts[0] * 0
                   for parts in zip(*updates, strict=True))
    effective = weights * scales
    retained = float(effective @ (context['risks'] / context['score_limit']) / effective.sum()) if effective.sum() > 0 else 0.
    return update, {'weights': weights.tolist(), 'scales': scales.tolist(), 'retained_risk': retained}


def adaptive_update(updates, context, action, *, norm_cap=1.0, maximum_amplification=2.0):
    risk_gate, mix, strength = (np.clip(action, -1, 1) + 1) / 2
    weights = (1 - risk_gate) + risk_gate * np.exp(-context['risks'])
    norms = context['norms']
    median_clip = np.minimum(1., norm_cap * np.median(norms) / np.maximum(norms, 1e-12))
    trusted = np.minimum(maximum_amplification, context['root_norm'] / np.maximum(norms, 1e-12))
    scales = (1 - strength) + strength * ((1 - mix) * median_clip + mix * trusted)
    return weighted_update(updates, context, weights, scales)
