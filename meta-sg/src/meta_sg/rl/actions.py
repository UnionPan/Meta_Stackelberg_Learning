"""Design: register action names and executable functions independently of TD3.
Data flow: selected policy controls + explicit fixed controls -> full action vector ->
client weights/scales and parameter delta. The selected names define policy width.
"""

from copy import deepcopy

import numpy as np

from meta_sg.defenses.adaptive import adaptive_update, weighted_update


def median_clip(updates, context, action, *, norm_cap=1.0):
    risk, strength = action
    return adaptive_update(updates, context, np.array([risk, -1., strength]), norm_cap=norm_cap)


def trusted_norm(updates, context, action, *, maximum_amplification=2.0):
    risk, strength = action
    return adaptive_update(updates, context, np.array([risk, 1., strength]), maximum_amplification=maximum_amplification)


def cnb(updates, context, action, *, threshold_min=.5, threshold_max=6.,
        norm_bound_min=.5, norm_bound_max=5., gate_temperature=2., scale_floors=(.05, .1, .0001)):
    unit = (np.clip(action, -1, 1) + 1) / 2
    thresholds = threshold_min + (threshold_max - threshold_min) * unit[:3]
    multiplier = norm_bound_min + (norm_bound_max - norm_bound_min) * unit[3]
    values = np.column_stack([1 - context['features'][:, 0], np.log(context['norms'] + 1e-12), context['features'][:, 2]])
    center = np.quantile(values, .5, axis=0, method='lower')
    scale = np.maximum(1.4826 * np.quantile(np.abs(values - center), .5, axis=0, method='lower'), scale_floors)
    anomalies = (values - center) / scale
    anomalies[:, 0] = np.maximum(-anomalies[:, 0], 0)
    anomalies[:, 1] = np.abs(anomalies[:, 1])
    anomalies[:, 2] = np.maximum(anomalies[:, 2], 0)
    gates = np.exp(-np.logaddexp(0., -gate_temperature * (thresholds - anomalies)))
    weights = gates.prod(1)
    scales = np.minimum(1., multiplier * np.median(context['norms']) / np.maximum(context['norms'], 1e-12))
    update, diagnostics = weighted_update(updates, context, weights, scales)
    return update, {**diagnostics, 'thresholds': thresholds.tolist(), 'norm_bound_multiplier': float(multiplier)}


ACTIONS = {
    'mixed_norm': (('risk', 'mix', 'strength'), adaptive_update),
    'median_clip': (('risk', 'strength'), median_clip),
    'trusted_norm': (('risk', 'strength'), trusted_norm),
    'cnb': (('cosine_threshold', 'norm_threshold', 'bias_threshold', 'norm_bound_multiplier'), cnb),
}


def resolve_action(spec, legacy_parameters=None):
    resolved = deepcopy(spec)
    names, _ = ACTIONS[resolved['name']]
    resolved['parameters'] = {**(legacy_parameters or {}), **resolved.get('parameters', {})}
    resolved.setdefault('controls', list(names))
    resolved.setdefault('fixed', {})
    controls, fixed = resolved['controls'], resolved['fixed']
    if (len(set(controls)) != len(controls) or set(controls) & set(fixed)
            or (set(controls) | set(fixed)) != set(names)):
        raise ValueError('action controls and fixed values must partition the component names without duplicates')
    return resolved


def build_action(spec):
    names, function = ACTIONS[spec['name']]
    controls = tuple(spec['controls'])

    def apply(updates, context, action):
        values = {**spec['fixed'], **dict(zip(controls, action, strict=True))}
        full_action = np.asarray([values[name] for name in names], dtype=np.float32)
        return function(updates, context, full_action, **spec['parameters'])

    return controls, apply
