"""Deterministic patch triggers for tensor image inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch


@dataclass(frozen=True)
class PatchTrigger:
    row: int
    column: int
    height: int
    width: int
    value: float

    def __post_init__(self) -> None:
        coordinates = (self.row, self.column, self.height, self.width)
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in coordinates):
            raise TypeError('patch coordinates and dimensions must be integers')
        if self.row < 0 or self.column < 0:
            raise ValueError('patch row and column must be non-negative')
        if self.height <= 0 or self.width <= 0:
            raise ValueError('patch height and width must be positive')
        if not math.isfinite(float(self.value)):
            raise ValueError('patch value must be finite')

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError('patch trigger requires a Torch tensor image')
        if image.ndim not in (2, 3):
            raise ValueError('patch trigger requires a 2-D or 3-D image tensor')
        image_height, image_width = int(image.shape[-2]), int(image.shape[-1])
        row_end = self.row + self.height
        column_end = self.column + self.width
        if row_end > image_height or column_end > image_width:
            raise ValueError('patch trigger exceeds image bounds')
        if not (torch.is_floating_point(image) or torch.is_complex(image)):
            if self.value != int(self.value):
                raise ValueError('patch value is not representable by image dtype')
        result = image.clone()
        result[..., self.row:row_end, self.column:column_end] = self.value
        return result
