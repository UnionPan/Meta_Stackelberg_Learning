"""Design: evaluate targeted backdoor success separately from clean accuracy.
Data flow: test batches -> eligible source/non-target examples -> complete trigger
-> target predictions / eligible count. Already-target examples are excluded.
"""

import torch


@torch.no_grad()
def evaluate_asr(model, loader, *, trigger, target_label, device, source_label=None):
    model.to(device)
    model.eval()
    hits, examples_seen = 0, 0
    for images, labels in loader:
        eligible = labels.ne(target_label)
        if source_label is not None:
            eligible &= labels.eq(source_label)
        if not eligible.any():
            continue
        triggered = trigger(images[eligible]).to(device)
        hits += model(triggered).argmax(dim=1).eq(target_label).sum()
        examples_seen += int(eligible.sum())
    if examples_seen == 0:
        raise ValueError('ASR evaluation requires at least one eligible example')
    return {'asr': float(hits) / examples_seen, 'examples_seen': examples_seen}
