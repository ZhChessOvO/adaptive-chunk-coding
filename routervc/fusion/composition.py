"""Composition. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np
import routervc.fusion.feather as legacy


def weights(shape, control):
    """E does not veto G. Explicit protection and zero blend still veto writes."""
    from types import SimpleNamespace

    mask = legacy.weights(shape, control, SimpleNamespace(packets=[]))
    return mask * control["blend"]


def combine(enhanced, generated, alpha):
    return (
        np.rint(
            enhanced.astype(np.float32)
            + alpha[..., None]
            * (generated.astype(np.float32) - enhanced.astype(np.float32))
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
