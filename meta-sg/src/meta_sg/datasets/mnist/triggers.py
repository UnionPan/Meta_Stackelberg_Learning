"""Design: use the existing four MNIST DBA patches on normalized image batches.
Data flow: [B, 1, 28, 28] images -> selected patch pixels -> triggered copies.
None composes all patches. White is expressed in the model's normalized space.
"""

import torch

from meta_sg.datasets.mnist.data import MEAN, STD


DBA_PATTERNS = (
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 6), (0, 7), (0, 8), (0, 9)),
    ((3, 0), (3, 1), (3, 2), (3, 3)),
    ((3, 6), (3, 7), (3, 8), (3, 9)),
)


def dba_trigger(images: torch.Tensor, *, trigger_index: int | None = None) -> torch.Tensor:
    triggered = images.clone()
    white = images.new_tensor(1.0).sub(MEAN).div(STD)
    patterns = DBA_PATTERNS if trigger_index is None else (DBA_PATTERNS[trigger_index],)
    for pattern in patterns:
        for row, column in pattern:
            triggered[:, :, row, column] = white
    return triggered
