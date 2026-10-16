"""Receiver helpers. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from contextlib import nullcontext


def _precision(device):
    if str(device).startswith("cuda"):
        from routervc.utils.precision import codec_precision

        return codec_precision()
    return nullcontext()
