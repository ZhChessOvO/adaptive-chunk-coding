"""Regions. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations

GRID = 4


def region_slice(shape, index):
    if type(index) is not int or not 0 <= index < GRID * GRID:
        raise ValueError("invalid region ID")
    if len(shape) != 4 or shape[:2] != (1, 256) or min(shape[2:]) < GRID:
        raise ValueError("invalid compact latent shape")
    height, width = shape[2:]
    row, col = divmod(index, GRID)
    return (
        slice(None),
        slice(None),
        slice(row * height // GRID, (row + 1) * height // GRID),
        slice(col * width // GRID, (col + 1) * width // GRID),
    )
