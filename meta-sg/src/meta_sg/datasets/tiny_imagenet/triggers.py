"""Design: preserve the later Tiny ImageNet white 80-pixel backdoor pattern.
Data flow: normalized RGB batch -> one of four 2x10 patches or their union.
The normalization preset must match the dataset loader.
"""

from meta_sg.datasets.tiny_imagenet.data import NORMALIZATION


def dba_trigger(images, *, trigger_index=None, normalization='legacy'):
    mean, std = NORMALIZATION[normalization]
    white = (1 - images.new_tensor(mean)) / images.new_tensor(std)
    triggered = images.clone()
    groups = range(4) if trigger_index is None else (trigger_index,)
    for group in groups:
        row_start, column_start = (group // 2) * 4, (group % 2) * 12
        triggered[:, :, row_start:row_start + 2, column_start:column_start + 10] = white[None, :, None, None]
    return triggered
