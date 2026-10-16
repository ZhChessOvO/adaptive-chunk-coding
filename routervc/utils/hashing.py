"""Hashing. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import hashlib
import torch


def tensor_identity(value):
    value = value.detach().contiguous().cpu()
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    )
