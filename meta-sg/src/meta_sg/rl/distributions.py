"""Design: preserve the original training-frozen Histogram40 transformation.
Data flow: training C/N/B/Z matrices -> quantile edges and summary mean/std ->
JSON artifact -> frozen encoding. Flattening is bin-major, then feature.
Inference never fits boundaries or normalization on deployment/evaluation data.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = {'histogram40': 'histogram40'}


def distribution_features(context, *, signed_log_norm=False):
    features = np.asarray(context['features'], dtype=np.float32).copy()
    if signed_log_norm:
        features[:, 1] = [math.log(max(float(norm), np.finfo(np.float64).tiny)) for norm in context['norms']]
    return features


def summarize(values, variant, edges=None):
    if variant != 'histogram40':
        raise ValueError('only Histogram40 transforms are supported')
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape != (9, 4) or np.any(np.diff(edges, axis=0) < 0):
        raise ValueError('histogram40 needs frozen nondecreasing [9, 4] boundaries')
    result = np.stack([
        np.bincount(np.searchsorted(edges[:, column], values[:, column], side='right'), minlength=10) / len(values)
        for column in range(4)
    ], axis=1).reshape(-1)
    return result.astype(np.float32)


class DistributionTransform:
    def __init__(self, variant, mean, scale, edges=None, standardized=True):
        if variant != 'histogram40':
            raise ValueError('only Histogram40 transforms are supported')
        self.variant, self.standardized = variant, standardized
        self.mean, self.scale = np.array(mean, dtype=float, copy=True), np.array(scale, dtype=float, copy=True)
        if self.mean.shape != (40,) or self.scale.shape != (40,) or np.any(self.scale <= 0):
            raise ValueError('frozen mean and positive scale must match observation width')
        self.edges = None if edges is None else np.array(edges, dtype=float, copy=True)
        summarize(np.zeros((2, 4)), variant, self.edges)
        for array in (self.mean, self.scale, self.edges):
            if array is not None:
                array.setflags(write=False)

    @classmethod
    def fit(cls, training_clients, variant, *, standardized=True):
        clients = [np.asarray(values, dtype=np.float64) for values in training_clients]
        edges = np.quantile(np.concatenate(clients), np.arange(.1, 1, .1), axis=0)
        observations = np.stack([summarize(values, variant, edges) for values in clients])
        mean = observations.astype(float).mean(0) if standardized else np.zeros(observations.shape[1])
        scale = np.maximum(observations.astype(float).std(0), 1e-4) if standardized else np.ones(observations.shape[1])
        return cls(variant, mean, scale, edges, standardized)

    def encode(self, clients):
        return ((summarize(clients, self.variant, self.edges) - self.mean) / self.scale).astype(np.float32)

    def to_dict(self):
        return {'schema_version': 1, 'variant': self.variant, 'mean': self.mean.tolist(),
                'scale': self.scale.tolist(), 'edges': None if self.edges is None else self.edges.tolist(),
                'standardized': self.standardized, 'fit_scope': 'training contexts only'}

    @classmethod
    def from_dict(cls, saved):
        if saved['schema_version'] != 1:
            raise ValueError('only frozen-summary schema_version 1 is supported')
        return cls(**{key: saved[key] for key in ('variant', 'mean', 'scale', 'edges', 'standardized')})

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


class FrozenDistributionObservation:
    def __init__(self, name, *, transform_path, signed_log_norm=False):
        self.variant = COMPONENTS[name]
        self.path = Path(transform_path)
        self.signed_log_norm = signed_log_norm
        self.transform, self.saved = None, None

    def prepare(self):
        if self.transform is None:
            self.load_state_dict(json.loads(self.path.read_text(encoding='utf-8')))

    def __call__(self, context):
        self.prepare()
        features = distribution_features(context, signed_log_norm=self.signed_log_norm)
        return self.transform.encode(features)

    def state_dict(self):
        self.prepare()
        return deepcopy(self.saved)

    def load_state_dict(self, state):
        transform = DistributionTransform.from_dict(state)
        if transform.variant != self.variant:
            raise ValueError('observation component and saved distribution variant differ')
        feature_norm = state.get('feature_norm', 'as_supplied')
        expected = 'signed_log_norm' if self.signed_log_norm else 'absolute_log_norm_ratio'
        if feature_norm != 'as_supplied' and feature_norm != expected:
            raise ValueError('observation N feature definition differs from the fitted transform')
        self.saved, self.transform = deepcopy(state), transform
