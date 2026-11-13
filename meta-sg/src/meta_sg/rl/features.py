"""Design: retain raw C/N/B/Z features and robust within-round deviations.
Data flow: global/client parameters and trusted root -> cosine discrepancy,
absolute log norm ratio, mean-probability shift, root loss change -> risk scores.
Candidate probes use the same global BatchNorm buffers and deterministic root data.
"""

from copy import deepcopy

import numpy as np
import torch

from meta_sg.federated.state import apply_model_update, flatten_update


@torch.no_grad()
def root_metrics(model, loader, device):
    model.eval()
    loss, correct, count, probabilities = 0., 0, 0, None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
        correct += int(logits.argmax(1).eq(labels).sum())
        count += len(labels)
        total = logits.softmax(1).sum(0)
        probabilities = total if probabilities is None else probabilities + total
    return {'loss': loss / count, 'accuracy': correct / count}, (probabilities / count).cpu().numpy()


@torch.no_grad()
def round_context(model, updates, root_update, loader, device, *,
                  scale_floors=(.05, .1, .0001, .005), risk_threshold=2.5, score_limit=8.):
    before, baseline_probabilities = root_metrics(model, loader, device)
    probe = deepcopy(model)
    base = deepcopy(model.state_dict())
    root_vector = flatten_update(root_update)
    root_norm = float(root_vector.norm())
    features, norms = [], []
    for update in updates:
        vector = flatten_update(update)
        norm = float(vector.norm())
        cosine = float(vector @ root_vector) / (norm * root_norm) if norm * root_norm > 0 else 0.
        probe.load_state_dict(base)
        apply_model_update(probe, update)
        after, probabilities = root_metrics(probe, loader, device)
        features.append([1 - np.clip(cosine, -1, 1), abs(np.log(norm / root_norm)) if norm * root_norm > 0 else 0.,
                         float(np.abs(probabilities - baseline_probabilities).max()), after['loss'] - before['loss']])
        norms.append(norm)
    features, norms = np.asarray(features, dtype=np.float64), np.asarray(norms)
    center = np.median(features, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(features - center), axis=0), scale_floors)
    scores = (features - center) / scale
    risks = np.maximum(scores - risk_threshold, 0).sum(1).clip(0, score_limit)
    return {'features': features, 'risks': risks, 'norms': norms,
            'root_norm': root_norm, 'before': before, 'score_limit': score_limit}
