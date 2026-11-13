"""Design: preserve the existing four CIFAR-10 DBA patterns and black pixels.
Data flow: normalized RGB batches -> one patch or their union -> triggered copy.
"""

from meta_sg.datasets.cifar10.data import MEAN, STD


DBA_PATTERNS = (
    ((5, 5), (6, 5), (7, 5), (8, 5)),
    ((9, 5), (10, 5), (11, 5)),
    ((8, 2), (8, 3), (8, 4), (8, 5), (8, 6)),
    ((8, 5), (8, 6), (8, 7), (8, 8)),
)


def dba_trigger(images, *, trigger_index=None):
    triggered = images.clone()
    black = -images.new_tensor(MEAN) / images.new_tensor(STD)
    patterns = DBA_PATTERNS if trigger_index is None else (DBA_PATTERNS[trigger_index],)
    for pattern in patterns:
        for row, column in pattern:
            triggered[:, :, row, column] = black
    return triggered
