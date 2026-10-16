"""Precision. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from contextlib import contextmanager
import torch
from routervc.utils.torch import configure_torch


@contextmanager
def codec_precision():
    backends = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "deterministic"),
    )
    saved = [(obj, key, getattr(obj, key)) for obj, key in backends]
    configure_torch()
    try:
        yield
    finally:
        for obj, key, value in saved:
            setattr(obj, key, value)
