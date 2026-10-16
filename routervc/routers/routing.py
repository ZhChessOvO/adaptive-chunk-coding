"""Routing. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np

SEED = 20261008


def coverage(parsed):
    """Packet-derived temporal coverage; I is not an E-capable frame."""
    info = parsed["info"]
    denominator = info["frames"] - 1
    result = np.zeros(16, np.float32)
    for chunk, region in parsed["packets"]:
        result[region] += min(8, denominator - 8 * chunk) / denominator
    return result


def control(shape, hashes, region, seed=SEED):
    from routervc.routers.inputs import grid_rois

    if type(region) is not int or not 0 <= region < 16 or shape[0] != 17:
        raise ValueError("invalid single G region")
    roi = grid_rois(*shape[1:3])[region]
    x, y, w, h = roi
    x0, y0 = (max(0, x - 64), max(0, y - 64))
    cw, ch = (min(shape[2], x + w + 64) - x0, min(shape[1], y + h + 64) - y0)
    if min(w, h) <= 32 or x0 % 8 or y0 % 8 or cw % 16 or ch % 16:
        raise ValueError("G processing geometry is unaligned")
    return dict(
        hashes,
        seed=seed + 65536 * region,
        strength=1.0,
        blend=1.0,
        window=17,
        stride=8,
        context=64,
        feather=16,
        processing_scale=1,
        protect=[],
        generate=[[0, 17, *roi]],
    )
