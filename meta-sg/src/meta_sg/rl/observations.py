"""Design: observation functions consume only deployment-visible round context.
Data flow: raw client C/N/B/Z features -> configurable summaries -> float32 vector.
Register another function in OBSERVATIONS to replace the summary without editing TD3.
"""

from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np

from meta_sg.rl.distributions import COMPONENTS, FrozenDistributionObservation


def quantiles(context, *, quantiles=(.5, .9), columns=(0, 1, 2, 3)):
    return np.quantile(context['features'][:, columns], quantiles, axis=0).ravel().astype(np.float32)


def compact8(context):
    return quantiles(context)


def statistics(context, *, columns=(0, 1, 2, 3)):
    values = context['features'][:, columns]
    return np.concatenate([values.mean(0), values.std(0), values.min(0), values.max(0)]).astype(np.float32)


def histogram(context, *, bins=10, ranges=((0, 2), (0, 5), (0, 1), (-1, 1))):
    features = context['features']
    return np.concatenate([
        np.histogram(np.clip(features[:, column], low, high), bins=bins, range=(low, high))[0] / len(features)
        for column, (low, high) in enumerate(ranges)
    ]).astype(np.float32)


OBSERVATIONS = {'compact8': compact8, 'quantiles': quantiles, 'statistics': statistics, 'histogram': histogram}


def frozen_distribution(context, *, name, transform_path, signed_log_norm=False):
    return FrozenDistributionObservation(name, transform_path=transform_path, signed_log_norm=signed_log_norm)(context)


OBSERVATIONS.update({name: partial(frozen_distribution, name=name) for name in COMPONENTS})


def resolve_observation(spec):
    resolved = deepcopy(spec)
    resolved.setdefault('parameters', {})
    if resolved['name'] in COMPONENTS:
        path = resolved['parameters']['transform_path']
        resolved['parameters']['transform_path'] = str(Path(path).expanduser().resolve())
        resolved['parameters'].setdefault('signed_log_norm', False)
    return resolved
