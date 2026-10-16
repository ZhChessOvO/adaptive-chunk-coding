"""Grid. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from numbers import Integral
import numpy as np

GRID_SIDE = 4


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported range")
    return value


def grid_rois(height, width):
    """Cover the complete frame with aligned cells, without resize or cropping."""
    height = _integer(height, "height", 64)
    width = _integer(width, "width", 64)
    if height % 8 or width % 8:
        raise ValueError("height and width must be multiples of eight")
    ys = [8 * (i * (height // 8) // GRID_SIDE) for i in range(GRID_SIDE + 1)]
    xs = [8 * (i * (width // 8) // GRID_SIDE) for i in range(GRID_SIDE + 1)]
    return [
        [xs[x], ys[y], xs[x + 1] - xs[x], ys[y + 1] - ys[y]]
        for y in range(GRID_SIDE)
        for x in range(GRID_SIDE)
    ]
