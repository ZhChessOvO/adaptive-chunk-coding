"""Feather. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np


def region_slice(region):
    start, count, x, y, w, h = region
    return np.s_[start : start + count, y : y + h, x : x + w]


def weights(shape, control, inner):
    """Nonnegative inner feather; generation cannot write protection/E pixels."""
    out = np.zeros(shape[:3], np.float32)
    feather = control["feather"]
    for r in control["generate"]:
        _, _, _, _, w, h = r
        xx, yy = (np.arange(w), np.arange(h))
        distance = np.minimum(
            np.minimum(xx, w - 1 - xx)[None, :], np.minimum(yy, h - 1 - yy)[:, None]
        )
        alpha = (
            np.minimum(distance / max(feather, 1), 1) if feather else np.ones((h, w))
        )
        out[region_slice(r)] = alpha
    for r in control["protect"]:
        out[region_slice(r)] = 0
    for p in inner.packets:
        out[region_slice([p.meta["start"], p.meta["count"], *p.meta["roi"]])] = 0
    return out
